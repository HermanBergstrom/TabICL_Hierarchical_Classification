"""
Simple TabICL baseline for flat multiclass text classification.

Predicts the leaf label without fine-tuning, using:
- Pre-extracted embeddings (if available)
- Or on-the-fly BERT extraction (frozen backbone)

Usage:
    # Use pre-extracted embeddings (fast)
    python tabicl_model.py --dataset wos --embedding-dir ./embeddings_cache/wos

    # Or extract embeddings on the fly
    python tabicl_model.py --dataset wos
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import MultiLabelBinarizer
try:
    from cuml.decomposition import PCA
except ImportError:
    PCA = sklearnPCA
from tabicl import TabICLClassifier
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Add repo to path
import sys
import os
repo_root = Path(__file__).parent / "dsi-nlp-publib" / "htc-survey-24"
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from src.dataset_tools.dataset_manager import DatasetManager
from src.utils.generic_functions import load_yaml
from src.utils.metrics import h_multilabel_precision_recall_fscore
from eval_metrics import compute_flat_metrics
from src.utils.text_utilities.preprocess import text_cleanup
from preprocess_taxonomy import load_taxonomy, TAX_PATHS, build_parent_map
from plot_confusion_matrix import save_cm_with_taxonomy_blocks, save_block_cms


class TabICLFlat:
    """Simple TabICL for flat multiclass text classification."""

    def __init__(
        self,
        dataset_name: str,
        embedding_dir: Optional[Path] = None,
        model_type: str = "bert",
        strategy: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 32,
        n_subsamples: Optional[int] = None,
        pca_dim: Optional[int] = 256,
        confusion_matrix_dir: Optional[Path] = None,
    ):
        """
        Args:
            dataset_name: Dataset name (wos, bugs, amazon, bgc)
            embedding_dir: Path to pre-extracted embeddings (overrides model_type/strategy)
            model_type: bert | electra | codebert — selects default embedding cache directory
            strategy: Pooling strategy used during extraction (e.g. concat_cls, mean_pool, cls-1).
                      Defaults to concat_cls for bert, mean_pool for electra/codebert.
            device: cuda or cpu
            n_subsamples: Subsample training to this many samples (None = use all)
            pca_dim: PCA dimension to apply before TabICL.
        """
        self.dataset_name = dataset_name
        self.device = device
        self.batch_size = batch_size
        self.n_subsamples = n_subsamples
        self.pca_dim = pca_dim
        self.confusion_matrix_dir = Path(confusion_matrix_dir) if confusion_matrix_dir else None

        if embedding_dir:
            self.embedding_dir = Path(embedding_dir)
        else:
            if strategy:
                effective_strategy = strategy
            elif model_type == "bert":
                effective_strategy = "concat_cls"
            elif model_type == "tfidf":
                effective_strategy = "tfidf_chi2768"
            else:
                effective_strategy = "mean_pool"
            self.embedding_dir = (
                Path(__file__).resolve().parent
                / f"precomputed_data/embeddings_cache_{model_type}_{effective_strategy}"
                / dataset_name
            )

        # Load config
        config_name = "amz" if dataset_name == "amazon" else dataset_name
        config_path = Path("config") / "BERT" / f"bert_{config_name}.yml"
        self.config = load_yaml(config_path)

        # Load dataset
        print(f"Loading {dataset_name} dataset...")
        self.dm = DatasetManager(dataset_name, self.config)
        self.targets = np.asarray(self.dm.labels_for_splitter)

        # Check if using pre-extracted embeddings (global or per-fold format)
        _has_global = self.embedding_dir and self.embedding_dir.exists() and (self.embedding_dir / "embeddings.npy").exists()
        _has_per_fold = self.embedding_dir and self.embedding_dir.exists() and (self.embedding_dir / "embeddings_train_fold_0.npy").exists()
        if _has_global or _has_per_fold:
            print(f"✓ Using pre-extracted embeddings from {self.embedding_dir}")
            self.use_cache = True
            self._load_cache()
        else:
            print(f"Extracting BERT embeddings on the fly...")
            self.use_cache = False
            self.embedding_dir = None
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModel.from_pretrained("bert-base-uncased").to(device)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        # Create CV splits
        self._create_splits()

        # Taxonomy for hF1 and taxonomy-ordered CM
        self._taxonomy = load_taxonomy(dataset_name)
        self._tax_path = str(repo_root / TAX_PATHS[dataset_name])
        all_tax_nodes = sorted({
            n
            for parent, children in self._taxonomy.items()
            for n in [parent] + children
            if n != "root"
        })
        self._mlb_hier = MultiLabelBinarizer(classes=all_tax_nodes)
        self._mlb_hier.fit([[]])
        self._parent_map = build_parent_map(self._taxonomy)

    def _load_cache(self):
        """Load pre-extracted embeddings and metadata."""
        with open(self.embedding_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.per_fold_embeddings = self.metadata.get("per_fold_embeddings", False)

        if self.per_fold_embeddings:
            # Embeddings are stored per-fold; load labels only
            self.embeddings = None
            cached_targets = np.load(self.embedding_dir / "labels.npy", allow_pickle=True)
            self.targets = cached_targets.astype(str)
        else:
            self.embeddings = np.load(self.embedding_dir / "embeddings.npy")
            cached_targets = np.load(self.embedding_dir / "labels.npy", allow_pickle=True)
            if cached_targets.ndim == 1 and cached_targets.shape[0] == self.embeddings.shape[0]:
                self.targets = cached_targets.astype(str)
            else:
                # Backward compatibility with older caches that stored multilabel matrices.
                self.targets = np.asarray(self.dm.labels_for_splitter)
        self.labels = self.targets

    def _create_splits(self):
        """Create CV splits (or load from cache if available), with optional subsampling."""
        if self.use_cache:
            self.splits = {}
            fold_idx = 0
            if getattr(self, "per_fold_embeddings", False):
                # Per-fold format: load pre-computed train/test embedding matrices directly
                self.fold_embeddings = {}
                while True:
                    train_emb_path = self.embedding_dir / f"embeddings_train_fold_{fold_idx}.npy"
                    test_emb_path  = self.embedding_dir / f"embeddings_test_fold_{fold_idx}.npy"
                    train_idx_path = self.embedding_dir / f"train_indices_fold_{fold_idx}.npy"
                    test_idx_path  = self.embedding_dir / f"test_indices_fold_{fold_idx}.npy"
                    if not train_emb_path.exists():
                        break
                    train_idx = np.load(train_idx_path)
                    test_idx  = np.load(test_idx_path)
                    if self.n_subsamples is not None:
                        keep = self._subsample(train_idx)
                        # Map global indices to local positions within the fold's train array
                        local_pos = np.searchsorted(train_idx, keep)
                        self.fold_embeddings[fold_idx] = {
                            "train": np.load(train_emb_path)[local_pos],
                            "test":  np.load(test_emb_path),
                        }
                        train_idx = keep
                    else:
                        self.fold_embeddings[fold_idx] = {
                            "train": np.load(train_emb_path),
                            "test":  np.load(test_emb_path),
                        }
                    self.splits[fold_idx] = {"train": train_idx, "test": test_idx}
                    fold_idx += 1
                print(f"Loaded {len(self.splits)} per-fold TF-IDF embedding folds from cache")
            else:
                # Global embeddings format: load split indices only
                while True:
                    train_path = self.embedding_dir / f"train_indices_fold_{fold_idx}.npy"
                    test_path = self.embedding_dir / f"test_indices_fold_{fold_idx}.npy"
                    if not train_path.exists() or not test_path.exists():
                        break
                    train_idx = np.load(train_path)
                    test_idx = np.load(test_path)

                    if self.n_subsamples is not None:
                        train_idx = self._subsample(train_idx)

                    self.splits[fold_idx] = {"train": train_idx, "test": test_idx}
                    fold_idx += 1
                print(f"Loaded {len(self.splits)} CV folds from cache")
            if self.n_subsamples:
                print(f"  Subsampled training to {self.n_subsamples} per fold")
        else:
            # Generate splits on the fly using baseline's exact splitter and seed
            self.splits = {}
            fold_idx = 0
            # Use exact same split generation as the baseline (which loads 'stratified_fold_seed' from config/random_seeds.yml)
            for train_idx, test_idx in self.dm.splitter.split(
                self.dm.data,
                self.dm.labels_for_splitter
            ):
                # Apply subsampling if requested
                if self.n_subsamples is not None:
                    train_idx = self._subsample(train_idx)

                self.splits[fold_idx] = {"train": train_idx, "test": test_idx}
                fold_idx += 1
            print(f"Generated {len(self.splits)} CV folds using baseline's splitter (seed: {self.dm.seeds['stratified_fold_seed']})")
            if self.n_subsamples:
                print(f"  Subsampled training to {self.n_subsamples} per fold")

    def _subsample(self, indices: np.ndarray) -> np.ndarray:
        """Subsample indices stratified by label."""
        if len(indices) <= self.n_subsamples:
            return indices

        # Stratified sampling based on labels
        labels_subset = np.asarray(self.targets)[indices]
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self.n_subsamples,
            random_state=42
        )
        subsample_idx, _ = next(splitter.split(
            np.zeros(len(labels_subset)),
            labels_subset
        ))

        return indices[subsample_idx]

    @staticmethod
    def _mean_emb(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (emb * mask).sum(dim=1) / mask.sum(dim=1)

    @staticmethod
    def _mean_emb_norm(emb: torch.Tensor, mask: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        masked_tensor = emb * mask
        num = masked_tensor.sum(dim=1)
        return num / (num.norm() if normalized else masked_tensor.norm(dim=2).sum())

    @staticmethod
    def _min_emb(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_tensor = emb.where(mask > 0, emb.max())
        return masked_tensor.min(dim=1)[0]

    @staticmethod
    def _max_emb(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_tensor = emb.where(mask > 0, emb.min())
        return masked_tensor.max(dim=1)[0]

    def _extract_embeddings(self, texts: list, batch_size: int = 32) -> np.ndarray:
        """Extract BERT embeddings for texts."""
        embeddings = []
        max_len = min(self.tokenizer.model_max_length, 512)
        strategy = self.config.get("CLF_STRATEGY", "concat_cls")
        num_layers = self.config.get("CLF_STRATEGY_NUM_LAYERS", 3)

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
                batch_texts = texts[i : i + batch_size]

                encoded = self.tokenizer(
                    batch_texts,
                    max_length=max_len,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states
                last_states = outputs.last_hidden_state

                # Mask [CLS] and [SEP]
                mask = attention_mask.detach().clone().float()
                mask[torch.arange(0, mask.size(0)).long(),
                      (mask.argmin(dim=1) - 1).clamp(0)] = 0.0
                mask[:, 0] = 0.0
                mask = mask.unsqueeze(-1)

                if strategy == "pool_cls":
                    embs = outputs.pooler_output
                elif strategy == "concat_pool_cls":
                    embs = torch.cat([self.model.pooler(e) for e in hidden_states[-num_layers:]], dim=1)
                elif strategy == "cls":
                    embs = last_states[:, 0]
                elif strategy == "concat_cls":
                    embs = torch.cat([e[:, 0] for e in hidden_states[-num_layers:]], dim=1)
                elif strategy == "avg_cls":
                    embs = torch.stack([e[:, 0] for e in hidden_states[-num_layers:]], dim=0).mean(dim=0)
                elif strategy == "mean_last":
                    embs = self._mean_emb(last_states, mask)
                elif strategy == "concat_mean_last":
                    layers_avg = [self._mean_emb(e, mask) for e in hidden_states[-num_layers:]]
                    embs = torch.cat(layers_avg, dim=1)
                elif strategy == "avg_mean_last":
                    layers_avg = [self._mean_emb(e, mask) for e in hidden_states[-num_layers:]]
                    embs = torch.stack(layers_avg, dim=0).mean(dim=0)
                elif strategy == "max_last":
                    embs = self._max_emb(last_states, mask)
                elif strategy == "concat_max_last":
                    layers_max = [self._max_emb(e, mask) for e in hidden_states[-num_layers:]]
                    embs = torch.cat(layers_max, dim=1)
                elif strategy == "avg_max_last":
                    layers_max = [self._max_emb(e, mask) for e in hidden_states[-num_layers:]]
                    embs = torch.stack(layers_max, dim=0).mean(dim=0)
                elif strategy == "max_min_last":
                    embs = torch.cat([self._max_emb(last_states, mask), self._min_emb(last_states, mask)], dim=1)
                elif strategy == "avg_max_min_last":
                    layers_avg = [torch.cat([self._max_emb(e, mask), self._min_emb(e, mask)], dim=1)
                                  for e in hidden_states[-num_layers:]]
                    embs = torch.stack(layers_avg, dim=0).mean(dim=0)
                elif strategy == "max_mean_last":
                    embs = torch.cat([self._max_emb(last_states, mask), self._mean_emb(last_states, mask)], dim=1)
                elif strategy == "avg_max_mean_last":
                    layers_avg = [torch.cat([self._max_emb(e, mask), self._mean_emb(e, mask)], dim=1)
                                  for e in hidden_states[-num_layers:]]
                    embs = torch.stack(layers_avg, dim=0).mean(dim=0)
                elif strategy == "mean_norm":
                    embs = self._mean_emb_norm(last_states, mask)
                elif strategy == "concat_mean_norm":
                    layers_avg = [self._mean_emb_norm(e, mask) for e in hidden_states[-num_layers:]]
                    embs = torch.cat(layers_avg, dim=1)
                elif strategy == "mean_normalized":
                    embs = self._mean_emb_norm(last_states, mask, normalized=True)
                elif strategy == "concat_mean_normalized":
                    layers_avg = [self._mean_emb_norm(e, mask, normalized=True) for e in hidden_states[-num_layers:]]
                    embs = torch.cat(layers_avg, dim=1)
                else:
                    raise ValueError(f"Unsupported CLF_STRATEGY '{strategy}'")

                embeddings.append(embs.cpu().numpy())

        return np.vstack(embeddings)

    def train_fold(self, fold_idx: int, n_estimators: int = 1) -> Dict:
        """Train and evaluate TabICL on a single fold."""
        split = self.splits[fold_idx]
        train_idx = split["train"]
        test_idx = split["test"]

        if self.use_cache:
            if getattr(self, "per_fold_embeddings", False):
                X_train = self.fold_embeddings[fold_idx]["train"]
                X_test  = self.fold_embeddings[fold_idx]["test"]
            else:
                X_train = self.embeddings[train_idx]
                X_test  = self.embeddings[test_idx]
            y_train = self.targets[train_idx]
            y_test = self.targets[test_idx]
        else:
            # Extract embeddings for this fold
            train_texts = [self.dm.data[i] for i in train_idx]
            test_texts = [self.dm.data[i] for i in test_idx]

            train_texts_clean = text_cleanup(
                train_texts,
                remove_garbage=self.config.get("REMOVE_GARBAGE_TEXT", False)
            )
            test_texts_clean = text_cleanup(
                test_texts,
                remove_garbage=self.config.get("REMOVE_GARBAGE_TEXT", False)
            )

            X_train = self._extract_embeddings(train_texts_clean, batch_size=self.batch_size)
            X_test = self._extract_embeddings(test_texts_clean, batch_size=self.batch_size)

            y_train = self.targets[train_idx]
            y_test = self.targets[test_idx]

        _t_fit = time.perf_counter()
        if self.pca_dim is not None and self.pca_dim > 0:
            print(f"  Applying PCA (dim={self.pca_dim})...")
            pca = PCA(n_components=self.pca_dim)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        print(f"  Flat classes: {len(np.unique(y_train))} unique leaf labels")

        # Train TabICL
        print(f"Training TabICL (n_estimators={n_estimators})...")
        clf = TabICLClassifier(n_estimators=n_estimators, random_state=42, device=self.device)
        clf.fit(X_train, y_train)
        fit_time = time.perf_counter() - _t_fit

        # Predict
        _t_eval = time.perf_counter()
        y_pred = clf.predict(X_test)

        # Flat metrics: ancestor-extended multi-hot, matching the baseline's label space
        flat_metrics = compute_flat_metrics(y_test, y_pred, self._parent_map, self._mlb_hier)

        # Hierarchical F1: extend leaf predictions with their taxonomy ancestors
        labels_true_hf1 = [[l] for l in y_test]
        labels_pred_hf1 = [[l] for l in y_pred]
        h_result = h_multilabel_precision_recall_fscore(
            labels_true_hf1, labels_pred_hf1,
            taxonomy_path=self._tax_path,
            encoder_dump_or_mapping=self._mlb_hier,
        )
        h_f1 = float(h_result.get("h_fscore", 0.0))
        eval_time = time.perf_counter() - _t_eval

        if self.confusion_matrix_dir is not None:
            save_cm_with_taxonomy_blocks(
                y_test, y_pred,
                self._taxonomy,
                self.confusion_matrix_dir / f"fold_{fold_idx}.png",
                title=f"Flat TabICL — {self.dataset_name} (fold {fold_idx})",
            )
            save_block_cms(
                y_test, y_pred,
                self._taxonomy,
                self.confusion_matrix_dir / f"fold_{fold_idx}",
                title_prefix=f"Flat TabICL — {self.dataset_name} (fold {fold_idx}) — ",
            )
            print(f"  Confusion matrix saved → {self.confusion_matrix_dir / f'fold_{fold_idx}.png'} "
                  f"(+ block CMs in fold_{fold_idx}/blocks/)")

        return {
            "fold":        fold_idx,
            "metrics":     flat_metrics,
            "h_f1":        h_f1,
            "n_train":     len(train_idx),
            "n_test":      len(test_idx),
            "predictions": y_pred,
            "fit_time":    fit_time,
            "eval_time":   eval_time,
        }

    def train_all_folds(self, n_estimators: int = 1) -> list:
        """Train and evaluate on all folds."""
        results = []
        for fold_idx in sorted(self.splits.keys()):
            result = self.train_fold(fold_idx, n_estimators=n_estimators)
            results.append(result)
        return results

    def print_summary(self, results: list):
        """Print results summary."""
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")

        accs  = [r["metrics"]["accuracy"] for r in results]
        f1s   = [r["metrics"]["macro_f1"]  for r in results]
        hf1s  = [r["h_f1"]                 for r in results]
        fit_eval_totals = [r["fit_time"] + r["eval_time"] for r in results]

        print(f"Accuracy:            {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"Macro F1:            {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"Hier. F1 (hF1):      {np.mean(hf1s):.4f} ± {np.std(hf1s):.4f}")
        print(f"Fit + Eval time (s): {np.mean(fit_eval_totals):.2f} ± {np.std(fit_eval_totals):.2f}")

        print(f"\nDataset: {self.dataset_name}")
        print(f"Num folds: {len(results)}")
        print(f"Using cache: {self.use_cache}")
        if self.use_cache:
            print(f"Cache dir: {self.embedding_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Simple flat TabICL for hierarchical text classification"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wos",
        choices=["wos", "bugs", "amazon", "bgc"]
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bert",
        choices=["bert", "electra", "codebert", "sbert", "tfidf"],
        help="Embedding model type"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Pooling strategy used during extraction (e.g. concat_cls, mean_pool, cls-1). "
             "Defaults to concat_cls for bert, mean_pool for electra/codebert. "
             "Sets the cache dir to embeddings_cache_{model_type}_{strategy}/{dataset}."
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default=None,
        help="Explicit path to embedding cache (overrides --model-type and --strategy)."
    )
    parser.add_argument(
        "--n-subsamples",
        type=int,
        default=None,
        help="Subsample training to this many samples per fold (None = use all)"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1,
        help="TabICL ensemble size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"]
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (only used if not using cache)"
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=256,
        help="PCA dimension to apply before TabICL. Set to 0 to disable. (Defaults to 256)"
    )
    parser.add_argument(
        "--confusion-matrix-dir",
        type=str,
        default=None,
        help="Directory to save per-fold flat confusion matrix plots."
    )

    args = parser.parse_args()

    # Initialize model
    model = TabICLFlat(
        dataset_name=args.dataset,
        embedding_dir=args.embedding_dir,
        model_type=args.model_type,
        strategy=args.strategy,
        device=args.device,
        batch_size=args.batch_size,
        n_subsamples=args.n_subsamples,
        pca_dim=args.pca_dim,
        confusion_matrix_dir=args.confusion_matrix_dir,
    )

    # Train on all folds
    print(f"\nTraining with n_estimators={args.n_estimators}...")
    results = model.train_all_folds(n_estimators=args.n_estimators)

    # Print summary
    model.print_summary(results)


if __name__ == "__main__":
    main()
