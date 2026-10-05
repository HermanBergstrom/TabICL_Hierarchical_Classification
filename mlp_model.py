"""
Simple 2-layer MLP baseline for flat multiclass text classification.

Loads the same pre-extracted embeddings as tabicl_model.py.
No PCA is applied; the raw embeddings are fed directly to the MLP.

Model selection: a stratified validation split (val_fraction of training data)
is used for early stopping — training halts when validation loss has not
improved for `patience` consecutive epochs, and the best checkpoint is restored.

Usage:
    python mlp_model.py --dataset wos --model-type bert
    python mlp_model.py --dataset wos --embedding-dir ./precomputed_data/embeddings_cache_bert_concat_cls/wos
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from tabicl_model import TabICLFlat  # also sets up sys.path / os.chdir

from eval_metrics import compute_flat_metrics


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPFlat(TabICLFlat):
    """2-layer MLP for flat multiclass text classification (no PCA)."""

    def __init__(
        self,
        dataset_name: str,
        embedding_dir: Optional[Path] = None,
        model_type: str = "bert",
        strategy: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 32,
        n_subsamples: Optional[int] = None,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 200,
        train_batch_size: int = 256,
        val_fraction: float = 0.1,
        patience: int = 10,
    ):
        """
        Args:
            hidden_dim:        Width of each hidden layer.
            dropout:           Dropout probability applied after each hidden layer.
            lr:                Adam learning rate.
            epochs:            Maximum number of training epochs.
            train_batch_size:  Mini-batch size for training.
            val_fraction:      Fraction of each fold's training set held out for
                               early-stopping validation (stratified by class).
            patience:          Stop training after this many epochs without
                               improvement in validation loss; restore the best checkpoint.
        """
        super().__init__(
            dataset_name=dataset_name,
            embedding_dir=embedding_dir,
            model_type=model_type,
            strategy=strategy,
            device=device,
            batch_size=batch_size,
            n_subsamples=n_subsamples,
            pca_dim=0,
        )
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.val_fraction = val_fraction
        self.patience = patience

    def train_fold(self, fold_idx: int, **_) -> Dict:
        """Train and evaluate a 2-layer MLP on a single fold."""
        split = self.splits[fold_idx]
        train_idx = split["train"]
        test_idx  = split["test"]

        if self.use_cache:
            if getattr(self, "per_fold_embeddings", False):
                X_train = self.fold_embeddings[fold_idx]["train"]
                X_test  = self.fold_embeddings[fold_idx]["test"]
            else:
                X_train = self.embeddings[train_idx]
                X_test  = self.embeddings[test_idx]
            y_train = self.targets[train_idx]
            y_test  = self.targets[test_idx]
        else:
            raise RuntimeError(
                "MLPFlat requires pre-extracted embeddings. "
                "Run extract_embeddings_cache.py first, or pass --embedding-dir."
            )

        classes = sorted(np.unique(y_train))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        y_train_int = np.array([class_to_idx[c] for c in y_train], dtype=np.int64)

        print(f"  Flat classes: {len(classes)} unique leaf labels")

        # --- Validation split for early stopping ---
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_fraction, random_state=42)
        tr_idx, val_idx = next(sss.split(X_train, y_train_int))

        X_tr,  X_val  = X_train[tr_idx],     X_train[val_idx]
        y_tr,  y_val  = y_train_int[tr_idx],  y_train_int[val_idx]

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long),
            ),
            batch_size=self.train_batch_size,
            shuffle=True,
        )
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(self.device)

        mlp = _MLP(X_train.shape[1], self.hidden_dim, len(classes), self.dropout).to(self.device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0
        best_epoch    = 0

        print(f"Training MLP (max {self.epochs} epochs, hidden_dim={self.hidden_dim}, "
              f"patience={self.patience})...")
        _t_fit = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            mlp.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                criterion(mlp(X_batch), y_batch).backward()
                optimizer.step()

            mlp.eval()
            with torch.no_grad():
                val_loss = criterion(mlp(X_val_t), y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in mlp.state_dict().items()}
                best_epoch    = epoch
                no_improve    = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stopping at epoch {epoch} (best epoch {best_epoch}, "
                          f"val loss {best_val_loss:.4f})")
                    break
        else:
            print(f"  Completed {self.epochs} epochs (best epoch {best_epoch}, "
                  f"val loss {best_val_loss:.4f})")

        mlp.load_state_dict(best_state)
        fit_time = time.perf_counter() - _t_fit

        # --- Evaluation ---
        _t_eval = time.perf_counter()
        mlp.eval()
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test, dtype=torch.float32)),
            batch_size=1024,
            shuffle=False,
        )
        preds = []
        with torch.no_grad():
            for (X_batch,) in test_loader:
                preds.append(mlp(X_batch.to(self.device)).argmax(dim=1).cpu())
        y_pred = np.array([classes[i] for i in torch.cat(preds).numpy()])

        flat_metrics = compute_flat_metrics(y_test, y_pred, self._parent_map, self._mlb_hier)
        eval_time = time.perf_counter() - _t_eval

        return {
            "fold":        fold_idx,
            "metrics":     flat_metrics,
            "n_train":     len(train_idx),
            "n_test":      len(test_idx),
            "best_epoch":  best_epoch,
            "predictions": y_pred,
            "fit_time":    fit_time,
            "eval_time":   eval_time,
        }

    def train_all_folds(self) -> list:
        return [self.train_fold(fold_idx) for fold_idx in sorted(self.splits.keys())]

    def print_summary(self, results: list):
        print(f"\n{'='*60}")
        print("MLP RESULTS SUMMARY")
        print(f"{'='*60}")

        accs            = [r["metrics"]["accuracy"] for r in results]
        f1s             = [r["metrics"]["macro_f1"]  for r in results]
        fit_eval_totals = [r["fit_time"] + r["eval_time"] for r in results]
        best_epochs     = [r["best_epoch"] for r in results]

        print(f"Accuracy:            {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"Macro F1:            {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"Fit + Eval time (s): {np.mean(fit_eval_totals):.2f} ± {np.std(fit_eval_totals):.2f}")
        print(f"Best epoch:          {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}")

        print(f"\nDataset: {self.dataset_name}")
        print(f"Num folds: {len(results)}")
        if self.use_cache:
            print(f"Cache dir: {self.embedding_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="2-layer MLP flat baseline for hierarchical text classification"
    )
    parser.add_argument("--dataset", type=str, default="wos",
                        choices=["wos", "bugs", "amazon", "bgc"])
    parser.add_argument("--model-type", type=str, default="bert",
                        choices=["bert", "electra", "codebert", "sbert", "tfidf"],
                        help="Embedding model type (selects default cache dir)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Pooling strategy used during extraction (e.g. concat_cls, mean_pool). "
                             "Combined with --model-type to resolve the cache dir.")
    parser.add_argument("--embedding-dir", type=str, default=None,
                        help="Explicit path to embedding cache (overrides --model-type and --strategy).")
    parser.add_argument("--n-subsamples", type=int, default=None,
                        help="Subsample training to this many samples per fold (None = use all)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for on-the-fly BERT extraction (unused when cache is present)")
    parser.add_argument("--hidden-dim", type=int, default=512,
                        help="Hidden layer width (default: 512)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability (default: 0.1)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Maximum training epochs (default: 200)")
    parser.add_argument("--train-batch-size", type=int, default=256,
                        help="Mini-batch size for MLP training (default: 256)")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of training data held out for early-stopping validation "
                             "(default: 0.1)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience in epochs (default: 10)")

    args = parser.parse_args()

    model = MLPFlat(
        dataset_name=args.dataset,
        embedding_dir=args.embedding_dir,
        model_type=args.model_type,
        strategy=args.strategy,
        device=args.device,
        batch_size=args.batch_size,
        n_subsamples=args.n_subsamples,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        val_fraction=args.val_fraction,
        patience=args.patience,
    )

    results = model.train_all_folds()
    model.print_summary(results)


if __name__ == "__main__":
    main()
