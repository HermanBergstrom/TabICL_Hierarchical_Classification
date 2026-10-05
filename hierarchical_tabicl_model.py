"""
Hierarchical TabICL classifier for text classification.

Traverses the taxonomy tree top-down using batched BFS inference.
At each internal node all samples routed there are predicted in one
forward pass, then fanned out to their predicted children.

Classifier type per node is chosen automatically from precomputed support data:
  - Single-label node (every sample has exactly one true child): standard multi-class TabICL.
  - Multi-label node (any sample has multiple simultaneous true children): Binary Relevance —
    C independent binary TabICLs, one per child.

Evaluation tracks:
  - Flat accuracy / macro F1 over leaf-level labels (exact multilabel subset match).
  - Hierarchical F1 (hF1) and AHC, matching the baseline's metric computation.
  - Per-node accuracy: at each internal node, % of samples where at least one predicted
    child is among the sample's true children.
  - Per-level accuracy: per-node accuracy averaged over all nodes at each taxonomy depth.

Usage:
    python hierarchical_tabicl_model.py --dataset wos
    python hierarchical_tabicl_model.py --dataset bgc --min-class-balance-ratio 0.2

Prerequisites:
    1. python extract_embeddings_cache.py --dataset wos ...
    2. python preprocess_taxonomy.py --dataset wos
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from networkx import read_adjlist
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

try:
    from cuml.decomposition import PCA
except ImportError:
    PCA = sklearnPCA
from tabicl import TabICLClassifier
from tqdm import tqdm

repo_root = Path(__file__).parent / "dsi-nlp-publib" / "htc-survey-24"
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from src.utils.metrics import h_multilabel_precision_recall_fscore, set_to_label_distance

from preprocess_taxonomy import (
    TAX_PATHS,
    _subsample_per_class,
    build_parent_map,
    get_immediate_children_at_node,
    load_full_labels,
    load_taxonomy,
)


# ---------------------------------------------------------------------------
# Node classifier wrappers
# ---------------------------------------------------------------------------

class MultiClassNodeClassifier:
    """Wraps a single multi-class TabICLClassifier + PCA for one node."""

    def __init__(self, clf: TabICLClassifier, pca):
        self.clf = clf
        self.pca = pca

    def route(self, X_pca: np.ndarray, nmlnp: bool = False) -> List[List[str]]:
        """Return [[pred]] per sample, or [[]] if max prob < 0.5 and nmlnp=True."""
        if nmlnp:
            predictions = self.clf.predict(X_pca)
            proba = self.clf.predict_proba(X_pca)  # (n, n_classes)
            return [[pred] if row.max() >= 0.5 else [] for pred, row in zip(predictions, proba)]
        return [[p] for p in self.clf.predict(X_pca)]


class BinaryRelevanceNodeClassifier:
    """
    Wraps C independent binary TabICLClassifiers for one multi-label node.

    One binary classifier per child label. At inference, any child whose
    classifier exceeds the 0.5 threshold is selected. If none exceed the
    threshold, the single highest-scoring child is returned as a fallback.
    """

    def __init__(
        self,
        child_labels: List[str],
        clfs: List[TabICLClassifier],
        pca,
    ):
        self.child_labels = child_labels
        self.clfs = clfs
        self.pca = pca

    def predict_proba(self, X_pca: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_children) array of P(positive) for each child."""
        return np.stack(
            [clf.predict_proba(X_pca)[:, 1] for clf in self.clfs], axis=1
        )

    def route(self, X_pca: np.ndarray, threshold: float = 0.5, nmlnp: bool = False) -> List[List[str]]:
        """
        Return predicted children per sample.
        When nmlnp=True: return [] if all probs < threshold (sample stops routing here).
        When nmlnp=False: argmax fallback ensures at least one child is always selected.
        """
        proba = self.predict_proba(X_pca)  # (n, C)
        result = []
        for row in proba:
            chosen = [self.child_labels[i] for i, p in enumerate(row) if p >= threshold]
            if not chosen and not nmlnp:
                chosen = [self.child_labels[int(np.argmax(row))]]
            result.append(chosen)
        return result


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TabICLHierarchical:
    """
    Hierarchical in-context classifier that routes samples top-down through
    the taxonomy, instantiating one classifier per internal node.

    The classifier type at each node is determined automatically from the
    precomputed support data (no user flag needed):
      - MultiClassNodeClassifier  when every training sample at the node has
        exactly one true immediate child.
      - BinaryRelevanceNodeClassifier  when any training sample has multiple
        simultaneous true children (detected by duplicate indices in support_indices.npy).
    """

    def __init__(
        self,
        dataset_name: str,
        embedding_dir: str,
        node_data_dir: Optional[str] = None,
        device: str = "cuda",
        n_estimators: int = 1,
        pca_dim: int = 256,
        max_clf_cache: int = 64,
        eval_batch_size: int = 100_000,
        max_support_per_node: Optional[int] = None,
        min_class_balance_ratio: Optional[float] = None,
        nmlnp: Optional[bool] = None,
    ):
        """
        Args:
            dataset_name:             wos | bugs | amazon | bgc
            embedding_dir:            path to the embedding cache (contains embeddings.npy etc.)
            node_data_dir:            path to node_data/ (defaults to precomputed_data/node_data/{dataset})
            device:                   cuda or cpu
            n_estimators:             TabICL ensemble size
            pca_dim:                  max PCA components per node (clamped to data dimensionality)
            max_clf_cache:            max (node, fold) classifiers kept in memory at once
            eval_batch_size:          max test samples processed in one BFS pass
            max_support_per_node:     total support budget per node, distributed evenly across
                                      classes; None = use the full saved pool
            min_class_balance_ratio:  at binary-relevance nodes, subsample the majority class
                                      so the minority class makes up at least this fraction
                                      (e.g. 0.2 → at least 20% positive). None = no balancing.
            nmlnp:                    non-mandatory leaf node prediction; if True, a sample stops
                                      routing at a node when all children score below 0.5 (rather
                                      than falling back to the argmax child). Auto-detected as
                                      True for BGC; override explicitly if needed.
        """
        self.dataset_name = dataset_name
        self.embedding_dir = Path(embedding_dir)
        self.node_data_dir = (
            Path(node_data_dir) if node_data_dir
            else Path(__file__).resolve().parent / "precomputed_data/node_data" / dataset_name
        )
        self.device = device
        self.n_estimators = n_estimators
        self.pca_dim = pca_dim
        self.max_clf_cache = max_clf_cache
        self.eval_batch_size = eval_batch_size
        self.max_support_per_node = max_support_per_node
        self.min_class_balance_ratio = min_class_balance_ratio
        self.nmlnp = (dataset_name == "bgc") if nmlnp is None else nmlnp
        self._subsample_rng = np.random.RandomState(42)

        print(f"Loading embeddings from {self.embedding_dir}...")
        self.embeddings = np.load(self.embedding_dir / "embeddings.npy")

        self.taxonomy   = load_taxonomy(dataset_name)
        self.parent_map = build_parent_map(self.taxonomy)

        print("Loading full label paths...")
        self.full_labels = load_full_labels(dataset_name)

        self.folds = self._load_folds()
        print(f"Loaded {len(self.folds)} fold(s)")

        print(f"Using node data dir: {self.node_data_dir}")
        print(f"NMLNP mode: {self.nmlnp} "
              f"({'samples stop routing when all children < 0.5' if self.nmlnp else 'argmax fallback always routes to a child'})")
        if not self.node_data_dir.exists():
            raise FileNotFoundError(
                f"Node data directory not found: {self.node_data_dir}\n"
                f"Run: python preprocess_taxonomy.py --dataset {dataset_name}"
            )

        # --- Build label binarizers and taxonomy graph once ---
        # Flat MLB: covers all labels that appear in the dataset (for flat accuracy/F1)
        flat_label_set = sorted({lbl for sample in self.full_labels for lbl in sample})
        self._mlb_flat = MultiLabelBinarizer(classes=flat_label_set)
        self._mlb_flat.fit([[]])

        # Hierarchical MLB: covers ALL taxonomy nodes excluding root
        # (needed for hF1 ancestor extension — _extend_label_set adds internal nodes)
        all_tax_nodes = sorted({
            n
            for parent, children in self.taxonomy.items()
            for n in [parent] + children
            if n != "root"
        })
        self._mlb_hier = MultiLabelBinarizer(classes=all_tax_nodes)
        self._mlb_hier.fit([[]])

        # Undirected taxonomy graph for AHC shortest-path computation
        tax_path = str(Path(repo_root) / TAX_PATHS[dataset_name])
        self._tax_graph = read_adjlist(tax_path, nodetype=str)

        # LRU-style cache: {(node, fold_idx): MultiClassNodeClassifier | BinaryRelevanceNodeClassifier}
        self._clf_cache: Dict[Tuple[str, int], object] = {}

    # ------------------------------------------------------------------
    # Fold / data helpers
    # ------------------------------------------------------------------

    def _load_folds(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        folds = {}
        k = 0
        while True:
            tp = self.embedding_dir / f"train_indices_fold_{k}.npy"
            ep = self.embedding_dir / f"test_indices_fold_{k}.npy"
            if not tp.exists() or not ep.exists():
                break
            folds[k] = (np.load(tp), np.load(ep))
            k += 1
        return folds

    def _compute_node_depths(self) -> Dict[str, int]:
        """BFS from root to assign depth to every node (root=0)."""
        depths: Dict[str, int] = {"root": 0}
        queue = ["root"]
        while queue:
            node = queue.pop(0)
            for child in self.taxonomy.get(node, []):
                depths[child] = depths[node] + 1
                if child in self.taxonomy:
                    queue.append(child)
        return depths

    # ------------------------------------------------------------------
    # Support balancing (for binary relevance nodes)
    # ------------------------------------------------------------------

    @staticmethod
    def _balance_binary_support(
        X: np.ndarray,
        y_bin: np.ndarray,
        ratio: Optional[float],
        rng: np.random.RandomState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample the majority class so the minority class makes up at least `ratio`
        of the combined set. Both directions handled (too few positives or too few negatives).
        No-op if ratio is None or balance is already within bounds.
        """
        if ratio is None:
            return X, y_bin

        pos_idx = np.where(y_bin == 1)[0]
        neg_idx = np.where(y_bin == 0)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)

        if n_pos == 0 or n_neg == 0:
            return X, y_bin

        pos_frac = n_pos / (n_pos + n_neg)

        if pos_frac < ratio:
            # Positives too rare — subsample negatives
            target_neg = max(1, min(int(n_pos * (1 - ratio) / ratio), n_neg))
            keep = np.concatenate([pos_idx, rng.choice(neg_idx, target_neg, replace=False)])
        elif pos_frac > 1.0 - ratio:
            # Negatives too rare — subsample positives
            target_pos = max(1, min(int(n_neg * (1 - ratio) / ratio), n_pos))
            keep = np.concatenate([rng.choice(pos_idx, target_pos, replace=False), neg_idx])
        else:
            return X, y_bin

        rng.shuffle(keep)
        return X[keep], y_bin[keep]

    # ------------------------------------------------------------------
    # Classifier builders
    # ------------------------------------------------------------------

    def _build_multiclass_node(
        self,
        support_indices: np.ndarray,
        support_y: np.ndarray,
    ) -> MultiClassNodeClassifier:
        support_X = self.embeddings[support_indices].astype(np.float32)
        n_components = min(self.pca_dim, support_X.shape[0] - 1, support_X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(support_X)

        clf = TabICLClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            device=self.device,
        )
        clf.fit(X_pca, support_y)
        return MultiClassNodeClassifier(clf, pca)

    def _build_binary_relevance_node(
        self,
        support_indices: np.ndarray,
        support_y: np.ndarray,
    ) -> BinaryRelevanceNodeClassifier:
        support_X = self.embeddings[support_indices].astype(np.float32)
        n_components = min(self.pca_dim, support_X.shape[0] - 1, support_X.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(support_X)

        child_labels = sorted(np.unique(support_y))
        clfs = []
        for child in child_labels:
            y_bin = (support_y == child).astype(int)
            X_bal, y_bal = self._balance_binary_support(
                X_pca, y_bin, self.min_class_balance_ratio, self._subsample_rng
            )
            clf = TabICLClassifier(
                n_estimators=self.n_estimators,
                random_state=42,
                device=self.device,
            )
            clf.fit(X_bal, y_bal)
            clfs.append(clf)

        return BinaryRelevanceNodeClassifier(child_labels, clfs, pca)

    # ------------------------------------------------------------------
    # Lazy classifier loading with auto-dispatch
    # ------------------------------------------------------------------

    def _get_classifier(
        self, node: str, fold_idx: int, verbose: bool = False
    ) -> Optional[object]:
        """
        Lazily load, fit, and cache the node classifier for (node, fold).
        Auto-detects whether to use multiclass or binary relevance by checking
        whether any training sample has multiple true children at this node
        (i.e., its index appears more than once in support_indices.npy).
        Returns None if no precomputed node data exists.
        """
        key = (node, fold_idx)
        if key in self._clf_cache:
            return self._clf_cache[key]

        fold_dir = self.node_data_dir / node / f"fold_{fold_idx}"
        if not fold_dir.exists():
            return None

        support_indices = np.load(fold_dir / "support_indices.npy")
        support_y = np.load(fold_dir / "support_y.npy", allow_pickle=True).astype(str)

        if self.max_support_per_node is not None:
            n_classes = len(np.unique(support_y))
            per_class = max(1, self.max_support_per_node // n_classes)
            support_indices, support_y = _subsample_per_class(
                support_indices, support_y, per_class, self._subsample_rng
            )

        # Auto-detect: any duplicate index means a sample has multiple true children
        counts = np.bincount(support_indices, minlength=len(self.embeddings))
        node_is_multilabel = bool((counts[support_indices] > 1).any())

        n_classes = len(np.unique(support_y))
        if verbose:
            tag = "[fit binary-rel]" if node_is_multilabel else "[fit multiclass]"
            print(f"    {tag} {node}  ({n_classes} classes, {len(support_indices)} support)")

        if node_is_multilabel:
            clf_node = self._build_binary_relevance_node(support_indices, support_y)
        else:
            clf_node = self._build_multiclass_node(support_indices, support_y)

        if len(self._clf_cache) >= self.max_clf_cache:
            self._clf_cache.pop(next(iter(self._clf_cache)))

        self._clf_cache[key] = clf_node
        return clf_node

    # ------------------------------------------------------------------
    # BFS inference
    # ------------------------------------------------------------------

    def _bfs_pass(
        self,
        batch_idx: np.ndarray,
        fold_idx: int,
        node_stats: Dict[str, Dict],
        all_visited: Dict[int, Set[str]],
    ):
        """
        Run one BFS pass over a batch of sample indices.

        Both MultiClassNodeClassifier and BinaryRelevanceNodeClassifier expose
        the same route() interface returning List[List[str]], so the loop is
        unified. Samples can fan out to multiple children at multi-label nodes.

        Updates node_stats and all_visited in-place.
        """
        current_level: Dict[str, List[int]] = {"root": list(batch_idx)}
        depth = 0

        while current_level:
            next_level: Dict[str, List[int]] = defaultdict(list)
            nodes = list(current_level.items())

            for node, indices in tqdm(
                nodes,
                desc=f"  depth {depth:2d}  ({len(nodes)} node(s))",
                unit="node",
                leave=False,
            ):
                clf_node = self._get_classifier(node, fold_idx, verbose=False)
                if clf_node is None:
                    continue

                X = self.embeddings[np.array(indices)].astype(np.float32)
                X_pca = clf_node.pca.transform(X)
                routed = clf_node.route(X_pca, nmlnp=self.nmlnp)  # List[List[str]]

                for idx, predicted_children in zip(indices, routed):
                    all_visited[idx].update(predicted_children)

                    true_children = get_immediate_children_at_node(
                        self.full_labels[idx], node, self.parent_map
                    )
                    if true_children:
                        node_stats[node]["total"] += 1
                        if any(p in true_children for p in predicted_children):
                            node_stats[node]["correct"] += 1

                    for child in predicted_children:
                        if child in self.taxonomy:
                            next_level[child].append(idx)

            current_level = dict(next_level)
            depth += 1

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _compute_ahc(
        self,
        predicted_labels: List[List[str]],
        true_labels: List[List[str]],
    ) -> float:
        """Average Hierarchical Cost: mean shortest-path distance from predictions to true labels."""
        ahcs = []
        for preds, trues in zip(predicted_labels, true_labels):
            if not trues or not preds:
                ahcs.append(0.0)
                continue
            d = sum(
                set_to_label_distance(self._tax_graph, p, trues)
                for p in preds
            ) / len(trues)
            ahcs.append(d)
        return float(np.mean(ahcs))

    def evaluate_fold(self, fold_idx: int) -> Dict:
        """
        Evaluate on the test split for one fold using batched BFS inference.

        Returns flat accuracy, macro F1, hierarchical F1 (hF1), and AHC,
        computed to match the baseline's metric computation.
        """
        _, test_idx = self.folds[fold_idx]

        node_stats: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0})
        all_visited: Dict[int, Set[str]] = defaultdict(set)

        n_batches = max(1, int(np.ceil(len(test_idx) / self.eval_batch_size)))
        print(f"\nEvaluating fold {fold_idx} ({len(test_idx)} test samples, "
              f"{n_batches} BFS batch(es) of up to {self.eval_batch_size})...")

        for batch_start in tqdm(
            range(0, len(test_idx), self.eval_batch_size),
            desc=f"Fold {fold_idx} batches", unit="batch"
        ):
            batch = test_idx[batch_start : batch_start + self.eval_batch_size]
            self._bfs_pass(batch, fold_idx, node_stats, all_visited)

        # Build per-sample label lists
        predicted_labels = [list(all_visited.get(idx, set())) for idx in test_idx]
        true_labels      = [list(self.full_labels[idx])       for idx in test_idx]

        # --- Flat accuracy and macro F1 ---
        flat_label_set = set(self._mlb_flat.classes_)
        predicted_flat = [[l for l in preds if l in flat_label_set] for preds in predicted_labels]
        y_true_bin = self._mlb_flat.transform(true_labels)
        y_pred_bin = self._mlb_flat.transform(predicted_flat)

        flat_acc = float(accuracy_score(y_true_bin, y_pred_bin))
        _, _, flat_f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average="macro", zero_division=0
        )
        flat_f1 = float(flat_f1)

        # --- Hierarchical F1 (hF1) ---
        # h_multilabel_precision_recall_fscore extends label sets with ancestors
        # and computes macro F1 over the extended sets. We pass the MLB fitted on
        # all taxonomy nodes so it can encode internal node labels after extension.
        tax_path = str(Path(repo_root) / TAX_PATHS[self.dataset_name])
        h_result = h_multilabel_precision_recall_fscore(
            true_labels, predicted_labels,
            taxonomy_path=tax_path,
            encoder_dump_or_mapping=self._mlb_hier,
        )
        h_f1 = float(h_result.get("h_fscore", 0.0))

        # --- AHC ---
        ahc = self._compute_ahc(predicted_labels, true_labels)

        return {
            "fold":         fold_idx,
            "node_stats":   {k: dict(v) for k, v in node_stats.items()},
            "flat_accuracy": flat_acc,
            "flat_macro_f1": flat_f1,
            "h_f1":          h_f1,
            "ahc":           ahc,
            "n_test":        len(test_idx),
        }

    def evaluate_all_folds(self) -> List[Dict]:
        return [self.evaluate_fold(k) for k in sorted(self.folds.keys())]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self, results: List[Dict]):
        print(f"\n{'='*65}")
        print("HIERARCHICAL TabICL — RESULTS SUMMARY")
        print(f"{'='*65}")
        print(f"Dataset: {self.dataset_name}   Folds: {len(results)}")

        # ---- Main metrics ----
        accs  = [r["flat_accuracy"] for r in results]
        f1s   = [r["flat_macro_f1"] for r in results]
        hf1s  = [r["h_f1"]          for r in results]
        ahcs  = [r["ahc"]           for r in results]

        print(f"\n{'Metric':<20s}  {'Mean':>8s}  {'Std':>8s}")
        print("-" * 42)
        print(f"{'Flat Accuracy':<20s}  {np.mean(accs):>8.4f}  {np.std(accs):>8.4f}")
        print(f"{'Flat Macro F1':<20s}  {np.mean(f1s):>8.4f}  {np.std(f1s):>8.4f}")
        print(f"{'Hier. F1 (hF1)':<20s}  {np.mean(hf1s):>8.4f}  {np.std(hf1s):>8.4f}")
        print(f"{'AHC':<20s}  {np.mean(ahcs):>8.4f}  {np.std(ahcs):>8.4f}")

        # ---- Per-node accuracy ----
        all_nodes   = {node for r in results for node in r["node_stats"]}
        node_depths = self._compute_node_depths()

        node_acc: Dict[str, List[float]] = defaultdict(list)
        node_total: Dict[str, int] = defaultdict(int)
        for r in results:
            for node, s in r["node_stats"].items():
                if s["total"] > 0:
                    node_acc[node].append(s["correct"] / s["total"])
                    node_total[node] += s["total"]

        depth_accs: Dict[int, List[float]] = defaultdict(list)
        print(f"\n{'Per-Node Accuracy':42s}  {'depth':5s}  {'acc':6s}  {'n_eval':>7s}")
        print("-" * 65)
        for node in sorted(all_nodes, key=lambda n: (node_depths.get(n, -1), n)):
            accs_node = node_acc.get(node, [])
            if not accs_node:
                continue
            mean_acc = float(np.mean(accs_node))
            depth    = node_depths.get(node, -1)
            depth_accs[depth].append(mean_acc)
            print(f"  {node:40s}  {depth:5d}  {mean_acc:.4f}  {node_total[node]:7d}")

        print(f"\nPer-Level Accuracy (mean over nodes at each depth):")
        print("-" * 40)
        for depth in sorted(depth_accs):
            level_accs = depth_accs[depth]
            print(f"  Depth {depth}: {np.mean(level_accs):.4f}  ({len(level_accs)} node(s))")


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical TabICL for hierarchical text classification"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["wos", "bugs", "amazon", "bgc"])
    parser.add_argument("--model-type", type=str, default=None,
                        choices=["bert", "electra", "codebert", "bge", "sbert", "scibert"],
                        help="Embedding model type.")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Pooling strategy used during extraction (e.g. concat_cls, mean_pool, cls-1). "
                             "Defaults to concat_cls for bert, mean_pool for electra/codebert. "
                             "Combined with --model-type to resolve the cache dir.")
    parser.add_argument("--embedding-dir", type=str, default=None,
                        help="Explicit path to embedding cache (overrides --model-type and --strategy).")
    parser.add_argument("--node-data-dir", type=str, default=None,
                        help="Path to node_data/ directory "
                             "(defaults to precomputed_data/node_data/{dataset})")
    parser.add_argument("--n-estimators", type=int, default=1,
                        help="TabICL ensemble size")
    parser.add_argument("--pca-dim", type=int, default=256,
                        help="Max PCA components per node "
                             "(clamped to min(pca_dim, n_support-1, n_features))")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--max-clf-cache", type=int, default=64,
                        help="Max (node, fold) classifiers kept in memory")
    parser.add_argument("--eval-batch-size", type=int, default=100_000,
                        help="Max test samples processed in one BFS pass (default: 100 000)")
    parser.add_argument("--max-support-per-node", type=int, default=None,
                        help="Support budget per node, split evenly across classes. "
                             "Default: use the full saved pool.")
    parser.add_argument("--min-class-balance-ratio", type=float, default=None,
                        help="At binary-relevance nodes, subsample the majority class so "
                             "the minority class makes up at least this fraction "
                             "(e.g. 0.2 → at least 20%% positive). Default: no balancing.")
    parser.add_argument("--nmlnp", action="store_true", default=None,
                        help="Enable non-mandatory leaf node prediction: a sample stops routing "
                             "at a node when all children score below 0.5. "
                             "Auto-enabled for BGC; use --no-nmlnp to override.")
    parser.add_argument("--no-nmlnp", dest="nmlnp", action="store_false",
                        help="Disable NMLNP (force argmax routing to a leaf for every sample).")

    args = parser.parse_args()

    if args.embedding_dir is not None:
        embedding_dir = args.embedding_dir
    elif args.model_type is not None:
        effective_strategy = args.strategy or ("concat_cls" if args.model_type in ("bert", "scibert") else "mean_pool")
        embedding_dir = str(
            Path(__file__).resolve().parent
            / f"precomputed_data/embeddings_cache_{args.model_type}_{effective_strategy}"
            / args.dataset
        )
    else:
        candidates = [
            p
            for model, default_strat in [("bert", "concat_cls"), ("electra", "mean_pool"), ("codebert", "mean_pool"), ("bge", "mean_pool"), ("sbert", "mean_pool"), ("scibert", "concat_cls")]
            for p in [Path(__file__).resolve().parent / f"precomputed_data/embeddings_cache_{model}_{default_strat}" / args.dataset]
        ]
        embedding_dir = next(
            (str(p) for p in candidates if (p / "embeddings.npy").exists()), None
        )
        if embedding_dir is None:
            parser.error(
                f"No embedding cache found for '{args.dataset}'. "
                f"Tried: {[str(p) for p in candidates]}. "
                f"Run extract_embeddings_cache.py first, or pass --model-type / --embedding-dir."
            )
    print(f"Using embedding dir: {embedding_dir}")

    model = TabICLHierarchical(
        dataset_name=args.dataset,
        embedding_dir=embedding_dir,
        node_data_dir=args.node_data_dir,
        device=args.device,
        n_estimators=args.n_estimators,
        pca_dim=args.pca_dim,
        max_clf_cache=args.max_clf_cache,
        eval_batch_size=args.eval_batch_size,
        max_support_per_node=args.max_support_per_node,
        min_class_balance_ratio=args.min_class_balance_ratio,
        nmlnp=args.nmlnp,
    )

    results = model.evaluate_all_folds()
    model.print_summary(results)


if __name__ == "__main__":
    main()
