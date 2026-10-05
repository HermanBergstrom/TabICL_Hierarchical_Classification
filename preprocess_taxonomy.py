"""
Offline preprocessing for hierarchical TabICL.

For each internal node in the taxonomy tree, builds a per-fold support index set.
Outputs are saved to {output_dir}/node_data/ (defaults to {fold_dir}/node_data/).

Must be run once per dataset before hierarchical_tabicl_model.py.
No embeddings are stored here — only sample indices and labels.
This means preprocessing is independent of the embedding type (BERT/ELECTRA/etc.)
and only needs to be rerun if the dataset or fold splits change.

PCA and subsampling are applied at model load time for flexibility.

Usage:
    python preprocess_taxonomy.py --dataset wos

Output structure:
    {fold_dir}/node_data/
        {node_name}/
            metadata.json
            fold_{k}/
                support_indices.npy  # global sample indices into embeddings.npy
                support_y.npy        # immediate child labels (str)
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

repo_root = Path(__file__).parent / "dsi-nlp-publib" / "htc-survey-24"
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from src.dataset_tools import get_bgc_split_jsonl, get_amazon, read_bugs_jsonl, read_wos_dataset
from src.utils.generic_functions import read_dict_to_list_tree


TAX_PATHS: Dict[str, Path] = {
    "wos":    Path("data") / "taxonomies" / "wos_tax.txt",
    "bugs":   Path("data") / "taxonomies" / "bugs_tax.txt",
    "amazon": Path("data") / "taxonomies" / "amazon_tax.txt",
    "bgc":    Path("data") / "taxonomies" / "bgc_tax.txt",
}


def load_taxonomy(dataset_name: str) -> Dict[str, List[str]]:
    """Load taxonomy as {parent: [children]}. Only internal nodes appear as keys."""
    return read_dict_to_list_tree(TAX_PATHS[dataset_name])


def build_parent_map(taxonomy: Dict[str, List[str]]) -> Dict[str, str]:
    """Invert taxonomy to {child: parent} for every non-root node."""
    parent_map = {}
    for parent, children in taxonomy.items():
        for child in children:
            parent_map[child] = parent
    return parent_map


def get_immediate_children_at_node(
    sample_labels: List[str], node: str, parent_map: Dict[str, str]
) -> List[str]:
    """Return labels in sample_labels that are direct children of node."""
    return [lbl for lbl in sample_labels if parent_map.get(lbl) == node]


def get_internal_nodes_on_true_path(
    sample_labels: List[str], parent_map: Dict[str, str]
) -> Set[str]:
    """
    Return every internal node where this sample has at least one true immediate child.
    Equivalently, all nodes that appear as parent_map[l] for some l in sample_labels.
    """
    return {parent_map[lbl] for lbl in sample_labels if lbl in parent_map}


def load_full_labels(dataset_name: str) -> List[List[str]]:
    """
    Load the full label list for every sample. Ordering matches the embedding cache
    (same loader functions, same order). Fast: disk reads only, no BERT.

    For WOS/Bugs/Amazon: each element is [L1, L2] (a strict path).
    For BGC: each element is a flat multilabel list (labels from multiple levels).
    """
    if dataset_name == "wos":
        return [doc["labels"] for doc in read_wos_dataset()]
    elif dataset_name == "bugs":
        return [doc["labels"] for doc in read_bugs_jsonl()]
    elif dataset_name == "amazon":
        return [doc["labels"] for doc in get_amazon()]
    elif dataset_name == "bgc":
        train = get_bgc_split_jsonl("train")
        val   = get_bgc_split_jsonl("val")
        test  = get_bgc_split_jsonl("test")
        return [doc["labels"] for doc in train + val + test]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _subsample_per_class(
    X: np.ndarray,
    y: np.ndarray,
    max_per_class: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample to at most max_per_class entries per class, then shuffle."""
    keep = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


class TaxonomyPreprocessor:
    """
    Builds per-node support sets for all folds of a dataset.

    For each internal node in the taxonomy and each CV fold:
      1. Collect ALL training embeddings whose label path passes through this node.
         For BGC (multilabel), a single sample may contribute multiple (emb, child)
         entries if it belongs to multiple immediate children.
      2. Save raw support_X (no PCA) and support_y — the full pool.
         Subsampling and PCA are applied at model load time for flexibility.
    """

    def __init__(
        self,
        dataset_name: str,
        fold_dir: Path,
        output_dir: Optional[Path] = None,
        random_state: int = 42,
    ):
        """
        Args:
            dataset_name: wos | bugs | amazon | bgc
            fold_dir:     directory containing train_indices_fold_k.npy / test_indices_fold_k.npy
                          (i.e. any embedding cache dir — the actual embeddings.npy is not loaded)
            output_dir:   where to write node_data/; defaults to {fold_dir}/node_data
        """
        self.dataset_name = dataset_name
        self.fold_dir = Path(fold_dir)
        self.node_data_dir = Path(output_dir) / "node_data" if output_dir else Path(__file__).resolve().parent / "precomputed_data/node_data" / dataset_name
        self.rng = np.random.RandomState(random_state)

        print(f"Loading taxonomy for {dataset_name}...")
        self.taxonomy  = load_taxonomy(dataset_name)
        self.parent_map = build_parent_map(self.taxonomy)
        print(f"  {len(self.taxonomy)} internal nodes")

        print("Loading full label paths (fast, no BERT)...")
        self.full_labels = load_full_labels(dataset_name)
        print(f"  {len(self.full_labels)} samples")

        self.folds = self._load_folds()
        print(f"  {len(self.folds)} fold(s) found")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_folds(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        folds = {}
        k = 0
        while True:
            tp = self.fold_dir / f"train_indices_fold_{k}.npy"
            ep = self.fold_dir / f"test_indices_fold_{k}.npy"
            if not tp.exists() or not ep.exists():
                break
            folds[k] = (np.load(tp), np.load(ep))
            k += 1
        return folds

    def _build_node_support(
        self, node: str, train_idx: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Collect (global_index, child_label) pairs for all training samples that
        pass through this node. Returns (indices, y) arrays or None if insufficient.
        """
        idx_list, y_list = [], []
        for idx in train_idx:
            for child in get_immediate_children_at_node(
                self.full_labels[idx], node, self.parent_map
            ):
                idx_list.append(idx)
                y_list.append(child)

        if len(idx_list) == 0:
            return None

        indices = np.array(idx_list, dtype=np.int64)
        y = np.array(y_list)
        if len(np.unique(y)) < 2:
            return None  # can't train a classifier with a single class

        return indices, y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self):
        """Build and save node support sets for every internal node across all folds."""
        self.node_data_dir.mkdir(parents=True, exist_ok=True)

        for node in self.taxonomy:
            node_dir = self.node_data_dir / node
            node_dir.mkdir(parents=True, exist_ok=True)

            fold_meta = {}
            for fold_idx, (train_idx, _) in self.folds.items():
                raw = self._build_node_support(node, train_idx)
                if raw is None:
                    print(f"  {node} / fold {fold_idx}: skipped (< 2 classes in training set)")
                    continue

                indices, y = raw

                node_fold_dir = node_dir / f"fold_{fold_idx}"
                node_fold_dir.mkdir(parents=True, exist_ok=True)
                np.save(node_fold_dir / "support_indices.npy", indices)
                np.save(node_fold_dir / "support_y.npy", y)

                unique_cls, counts = np.unique(y, return_counts=True)
                fold_meta[fold_idx] = {
                    "n_support": int(len(y)),
                    "n_classes": int(len(unique_cls)),
                    "classes":   unique_cls.tolist(),
                    "counts":    counts.tolist(),
                }
                print(f"  {node:40s} fold {fold_idx}: "
                      f"{len(y):5d} support  "
                      f"{len(unique_cls):3d} classes")

            with open(node_dir / "metadata.json", "w") as f:
                json.dump({
                    "node":     node,
                    "children": self.taxonomy[node],
                    "folds":    fold_meta,
                }, f, indent=2)

        print(f"\nDone. Node data saved to {self.node_data_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline preprocessing for hierarchical TabICL"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["wos", "bugs", "amazon", "bgc"])
    parser.add_argument("--fold-dir", type=str, default=None,
                        help="Directory containing train_indices_fold_k.npy / test_indices_fold_k.npy. "
                             "Defaults to ./embeddings_cache_bert/{dataset} or "
                             "./precomputed_data/embeddings_cache_electra/{dataset} (first that exists). "
                             "The actual embeddings.npy is NOT loaded.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write node_data/. Defaults to {fold-dir}/node_data/.")
    args = parser.parse_args()

    if args.fold_dir is not None:
        fold_dir = Path(args.fold_dir)
    else:
        # Scan any embeddings_cache_* directory for fold indices
        candidates = sorted((Path(__file__).resolve().parent / "precomputed_data").glob(f"embeddings_cache_*/{args.dataset}"))
        fold_dir = next(
            (p for p in candidates if (p / "train_indices_fold_0.npy").exists()), None
        )
        if fold_dir is None:
            parser.error(
                f"No fold indices found for '{args.dataset}' in any embeddings_cache_* directory. "
                f"Run extract_embeddings_cache.py first, or pass --fold-dir."
            )
    print(f"Using fold dir: {fold_dir}")

    preprocessor = TaxonomyPreprocessor(
        dataset_name=args.dataset,
        fold_dir=fold_dir,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    preprocessor.preprocess()


if __name__ == "__main__":
    main()
