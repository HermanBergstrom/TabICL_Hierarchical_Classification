"""
Pre-extract and cache BERT embeddings using the same processing as the BERT baseline.

This script:
1. Loads datasets using DatasetManager (same as BERT baseline)
2. Applies identical text cleanup
3. Extracts BERT embeddings using the same tokenization settings
4. Caches raw embeddings (no PCA) for flexible downstream use
5. Stores metadata (splits, label binarizer, etc.) for reproducibility

PCA or other dimensionality reduction can be applied later during training if needed.

Usage:
    python extract_embeddings_cache.py --dataset wos --output-dir ./embeddings

The embeddings are saved with directory structure:
    embeddings/
    ├── wos/
    │   ├── embeddings.npy          # All embeddings (n_samples, embedding_dim) - raw, no PCA
    │   ├── indices_fold_0.npy      # Split indices for each fold
    │   ├── indices_fold_1.npy
    │   ├── metadata.json           # Dataset metadata, config, label binarizer path
    │   └── labels.npy              # Leaf labels used by TabICL
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Add the repo to path and change to repo directory
repo_root = Path(__file__).parent / "dsi-nlp-publib" / "htc-survey-24"
sys.path.insert(0, str(repo_root))
original_cwd = os.getcwd()
os.chdir(repo_root)

from src.dataset_tools.dataset_manager import DatasetManager
from src.dataset_tools import get_bgc_split_jsonl
from src.utils.generic_functions import load_yaml
from src.utils.text_utilities.preprocess import text_cleanup


class EmbeddingExtractor:
    """Extract and cache BERT or ELECTRA embeddings."""

    def __init__(
        self,
        dataset_name: str,
        config_path: Optional[Path] = None,
        bert_model: str = "bert-base-uncased",
        bert_strategy: Optional[str] = None,
        device: str = "cuda",
        model_type: str = "bert",
        electra_model: str = "google/electra-base-discriminator",
        codebert_model: str = "microsoft/codebert-base",
        codebert_strategy: str = "mean_pool",
        codebert_cls_layer: int = -1,
        bge_model: str = "BAAI/bge-base-en-v1.5",
        sbert_model: str = "sentence-transformers/all-mpnet-base-v2",
        scibert_model: str = "allenai/scibert_scivocab_uncased",
        scibert_strategy: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.device = device
        self.model_type = model_type

        # Load config
        if config_path is None:
            config_name = "amz" if dataset_name == "amazon" else dataset_name
            config_path = Path("config") / "BERT" / f"bert_{config_name}.yml"
        self.config = load_yaml(config_path)

        # Override BERT strategy from CLI; mean_pool is an alias for mean_last
        if bert_strategy is not None and model_type == "bert":
            resolved = "mean_last" if bert_strategy == "mean_pool" else bert_strategy
            self.config["CLF_STRATEGY"] = resolved

        # Load dataset
        print(f"Loading {dataset_name} dataset...")
        self.dm = DatasetManager(dataset_name, self.config)

        # BGC uses fixed train/val/test splits rather than CV.
        # Combine all splits into a single list so the rest of the pipeline
        # (extract_all_embeddings, create_cv_splits, save_cache) works uniformly.
        if dataset_name == "bgc":
            train_docs = get_bgc_split_jsonl("train")
            val_docs   = get_bgc_split_jsonl("val")
            test_docs  = get_bgc_split_jsonl("test")
            all_docs   = train_docs + val_docs + test_docs
            self.dm.data = [doc["text"] for doc in all_docs]
            self.dm.labels_for_splitter = [doc["labels"][-1] for doc in all_docs]
            self._bgc_n_train = len(train_docs) + len(val_docs)
        else:
            self._bgc_n_train = None

        # Store codebert-specific settings
        if model_type == "codebert":
            self.codebert_strategy = codebert_strategy
            self.codebert_cls_layer = codebert_cls_layer

        # Store scibert-specific settings and override strategy from CLI
        if model_type == "scibert":
            if scibert_strategy is not None:
                resolved = "mean_last" if scibert_strategy == "mean_pool" else scibert_strategy
                self.config["CLF_STRATEGY"] = resolved
            self.scibert_strategy = scibert_strategy

        # Initialize model
        if model_type == "electra":
            model_name = electra_model
            print(f"Loading ELECTRA model: {model_name}")
        elif model_type == "codebert":
            model_name = codebert_model
            print(f"Loading CodeBERT model: {model_name}")
        elif model_type == "bge":
            model_name = bge_model
            print(f"Loading BGE model: {model_name}")
        elif model_type == "sbert":
            model_name = sbert_model
            print(f"Loading SBERT model: {model_name}")
        elif model_type == "scibert":
            model_name = scibert_model
            print(f"Loading SciBERT model: {model_name}")
        else:
            model_name = bert_model
            print(f"Loading BERT model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(device)

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Store metadata
        if model_type == "electra":
            self.metadata = {
                "dataset": dataset_name,
                "model_type": "electra",
                "electra_model": electra_model,
                "clf_strategy": "mean_pool_no_cls",
                "remove_garbage": self.config.get("REMOVE_GARBAGE_TEXT", False),
                "target_label_type": "leaf_label",
                "created_at": dt.datetime.now().isoformat(),
            }
        elif model_type == "codebert":
            self.metadata = {
                "dataset": dataset_name,
                "model_type": "codebert",
                "codebert_model": codebert_model,
                "clf_strategy": codebert_strategy,
                "codebert_cls_layer": codebert_cls_layer if codebert_strategy == "cls" else None,
                "remove_garbage": self.config.get("REMOVE_GARBAGE_TEXT", False),
                "target_label_type": "leaf_label",
                "created_at": dt.datetime.now().isoformat(),
            }
        elif model_type == "bge":
            self.metadata = {
                "dataset": dataset_name,
                "model_type": "bge",
                "bge_model": bge_model,
                "clf_strategy": "mean_pool",
                "l2_normalized": True,
                "remove_garbage": self.config.get("REMOVE_GARBAGE_TEXT", False),
                "target_label_type": "leaf_label",
                "created_at": dt.datetime.now().isoformat(),
            }
        elif model_type == "sbert":
            self.metadata = {
                "dataset": dataset_name,
                "model_type": "sbert",
                "sbert_model": sbert_model,
                "clf_strategy": "mean_pool",
                "l2_normalized": True,
                "remove_garbage": self.config.get("REMOVE_GARBAGE_TEXT", False),
                "target_label_type": "leaf_label",
                "created_at": dt.datetime.now().isoformat(),
            }
        elif model_type == "scibert":
            self.metadata = {
                "dataset": dataset_name,
                "model_type": "scibert",
                "scibert_model": scibert_model,
                "clf_strategy": self.config.get("CLF_STRATEGY", "concat_cls"),
                "clf_strategy_num_layers": self.config.get("CLF_STRATEGY_NUM_LAYERS", 3),
                "remove_garbage": self.config.get("REMOVE_GARBAGE_TEXT", False),
                "target_label_type": "leaf_label",
                "created_at": dt.datetime.now().isoformat(),
            }
        else:
            self.metadata = {
                "dataset": dataset_name,
                "model_type": "bert",
                "bert_model": bert_model,
                "clf_strategy": self.config.get("CLF_STRATEGY", "concat_cls"),
                "clf_strategy_num_layers": self.config.get("CLF_STRATEGY_NUM_LAYERS", 3),
                "remove_garbage": self.config.get("REMOVE_GARBAGE_TEXT", False),
                "target_label_type": "leaf_label",
                "created_at": dt.datetime.now().isoformat(),
            }

        # Canonical strategy name used for default output directory naming
        if model_type == "electra":
            self.effective_strategy = "mean_pool"
        elif model_type == "codebert":
            if codebert_strategy == "cls":
                self.effective_strategy = f"cls{codebert_cls_layer}"
            else:
                self.effective_strategy = "mean_pool"
        elif model_type in ("bge", "sbert"):
            self.effective_strategy = "mean_pool"
        elif model_type == "scibert":
            s = self.config.get("CLF_STRATEGY", "concat_cls")
            self.effective_strategy = "mean_pool" if s == "mean_last" else s
        else:  # bert
            s = self.config.get("CLF_STRATEGY", "concat_cls")
            # Normalise mean_last → mean_pool so it matches the user-facing alias
            self.effective_strategy = "mean_pool" if s == "mean_last" else s

    def _apply_text_cleanup(self, texts: list) -> list:
        """Apply identical text cleanup as the BERT baseline."""
        return text_cleanup(
            texts,
            remove_garbage=self.config.get("REMOVE_GARBAGE_TEXT", False)
        )

    def _extract_embeddings_batch(
        self,
        texts: list,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Extract embeddings for a batch of texts."""
        embeddings = []
        max_len = min(self.tokenizer.model_max_length, 512)

        device_type = self.device.split(":")[0]  # "cuda" or "cpu"
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

                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    if self.model_type in ("bge", "sbert"):
                        outputs = self.model(input_ids, attention_mask=attention_mask)
                        last_states = outputs.last_hidden_state  # (B, L, D)
                        # Mean pool over all non-padding tokens (including [CLS]),
                        # which matches how BGE and SBERT models were trained.
                        mask = attention_mask.float().unsqueeze(-1)  # (B, L, 1)
                        embs = (last_states * mask).sum(dim=1) / mask.sum(dim=1)
                        # L2 normalize embeddings (standard for sentence transformers)
                        embs = F.normalize(embs, p=2, dim=1)
                    elif self.model_type == "electra":
                        outputs = self.model(input_ids, attention_mask=attention_mask)
                        last_states = outputs.last_hidden_state  # (B, L, D)

                        # Mean pool over real tokens, excluding [CLS] at position 0
                        mask = attention_mask.float().unsqueeze(-1)  # (B, L, 1)
                        mask[:, 0] = 0.0  # exclude [CLS]
                        embs = (last_states * mask).sum(dim=1) / mask.sum(dim=1)
                    elif self.model_type == "codebert":
                        need_hidden = self.codebert_strategy == "cls"
                        outputs = self.model(
                            input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=need_hidden,
                        )
                        if self.codebert_strategy == "mean_pool":
                            last_states = outputs.last_hidden_state  # (B, L, D)
                            mask = attention_mask.float().unsqueeze(-1)  # (B, L, 1)
                            mask[:, 0] = 0.0  # exclude [CLS]
                            embs = (last_states * mask).sum(dim=1) / mask.sum(dim=1)
                        else:  # cls
                            # hidden_states: tuple of (n_layers+1,) tensors, index 0 = embeddings
                            embs = outputs.hidden_states[self.codebert_cls_layer][:, 0]
                    elif self.model_type == "scibert" or self.model_type == "bert":
                        outputs = self.model(
                            input_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True
                        )
                        hidden_states = outputs.hidden_states  # 13 layers for BERT
                        last_states = outputs.last_hidden_state

                        strategy = self.config.get("CLF_STRATEGY", "concat_cls")
                        num_layers = self.config.get("CLF_STRATEGY_NUM_LAYERS", 3)

                        # Prepare mask for pooling (exclude [CLS], [SEP], and padding)
                        mask = attention_mask.detach().clone().float()
                        mask[input_ids == self.tokenizer.cls_token_id] = 0.0  # [CLS]
                        mask[input_ids == self.tokenizer.sep_token_id] = 0.0  # [SEP]
                        mask = mask.unsqueeze(-1)

                        if strategy == "concat_cls":
                            embs = torch.cat([e[:, 0] for e in hidden_states[-num_layers:]], dim=1)
                        elif strategy == "mean_last":
                            embs = (last_states * mask).sum(dim=1) / mask.sum(dim=1)
                        elif strategy == "concat_mean_last":
                            layers_avg = [(e * mask).sum(dim=1) / mask.sum(dim=1) for e in hidden_states[-num_layers:]]
                            embs = torch.cat(layers_avg, dim=1)
                        elif strategy == "cls":
                            embs = last_states[:, 0]
                        elif strategy == "pool_cls":
                            embs = outputs.pooler_output
                        else:
                            print(f"[warning] Unknown strategy '{strategy}', falling back to concat_cls")
                            embs = torch.cat([e[:, 0] for e in hidden_states[-num_layers:]], dim=1)

                embeddings.append(embs.cpu().half().numpy())

        return np.vstack(embeddings)

    def extract_all_embeddings(self, batch_size: int = 32) -> np.ndarray:
        """Extract embeddings for entire dataset (no PCA - cache raw embeddings)."""
        print(f"\nApplying text cleanup (remove_garbage={self.config.get('REMOVE_GARBAGE_TEXT', False)})...")
        cleaned_texts = self._apply_text_cleanup(self.dm.data)

        model_label = {
            "electra": "ELECTRA", "codebert": "CodeBERT",
            "bge": "BGE", "sbert": "SBERT", "scibert": "SciBERT",
        }.get(self.model_type, "BERT")
        print(f"\nExtracting {model_label} embeddings...")
        embeddings = self._extract_embeddings_batch(cleaned_texts, batch_size=batch_size)

        print(f"Embeddings shape: {embeddings.shape}")
        if self.model_type in ("bge", "sbert"):
            print(f"Strategy: mean_pool (final layer, all non-padding tokens including [CLS])")
            print(f"✓ L2 normalized (matches sentence-transformers standard)")
        elif self.model_type == "electra":
            print(f"Strategy: mean_pool_no_cls (final layer, excluding [CLS] and padding)")
        elif self.model_type == "codebert":
            if self.codebert_strategy == "mean_pool":
                print(f"Strategy: mean_pool (final layer, excluding [CLS] and padding)")
            else:
                print(f"Strategy: cls (layer {self.codebert_cls_layer})")
        elif self.model_type == "scibert":
            strategy = self.config.get("CLF_STRATEGY", "concat_cls")
            num_layers = self.config.get("CLF_STRATEGY_NUM_LAYERS", 3)
            print(f"Strategy: {strategy}, Num layers: {num_layers}")
        else:
            print(f"Strategy: {self.config.get('CLF_STRATEGY', 'concat_cls')}, "
                  f"Num layers: {self.config.get('CLF_STRATEGY_NUM_LAYERS', 3)}")
        print(f"✓ Caching raw embeddings (no PCA applied - can be applied later)")

        self.metadata["pca_applied"] = False

        return embeddings

    def create_cv_splits(
        self,
        embeddings: np.ndarray,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Create CV split indices. BGC uses a single fixed split; all others use RepeatedStratifiedKFold."""
        splits = {}

        if self.dataset_name == "bgc":
            n_train = self._bgc_n_train
            n_total = len(embeddings)
            train_idx = np.arange(n_train)
            test_idx  = np.arange(n_train, n_total)
            splits[0] = (train_idx, test_idx)
            print(f"\nBGC fixed split: train+val={len(train_idx)}, test={len(test_idx)}")
            self.metadata.update({
                "n_folds": 1, "n_splits": 1, "n_repeats": 1,
                "stratified_fold_seed": None, "total_samples": n_total,
            })
        else:
            print(f"\nCreating CV splits ({self.config.get('stratifiedCV', 3)}-fold × {self.config.get('n_repeats', 2)} repeats)...")
            print(f"Using exactly the baseline random seed: {self.dm.seeds['stratified_fold_seed']}")
            fold_idx = 0
            for train_idx, test_idx in self.dm.splitter.split(
                self.dm.data, self.dm.labels_for_splitter
            ):
                splits[fold_idx] = (train_idx, test_idx)
                print(f"  Fold {fold_idx}: train={len(train_idx)}, test={len(test_idx)}")
                fold_idx += 1
            self.metadata["n_folds"] = len(splits)
            self.metadata["n_splits"] = self.config.get("stratifiedCV", 3)
            self.metadata["n_repeats"] = self.config.get("n_repeats", 2)
            self.metadata["stratified_fold_seed"] = self.dm.seeds["stratified_fold_seed"]
            self.metadata["total_samples"] = len(embeddings)

        return splits

    def save_cache(
        self,
        output_dir: Path,
        embeddings: np.ndarray,
        splits: Dict[int, Tuple[np.ndarray, np.ndarray]],
    ):
        """Save embeddings, splits, and metadata to disk."""
        output_dir = Path(output_dir) / self.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving to {output_dir}")

        # Save embeddings
        embeddings_path = output_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        print(f"  ✓ Embeddings: {embeddings_path} ({embeddings.nbytes / 1e9:.2f} GB)")

        # Save splits
        for fold_idx, (train_idx, test_idx) in splits.items():
            train_path = output_dir / f"train_indices_fold_{fold_idx}.npy"
            test_path = output_dir / f"test_indices_fold_{fold_idx}.npy"
            np.save(train_path, train_idx)
            np.save(test_path, test_idx)
        print(f"  ✓ Split indices: {len(splits)} folds")

        # Save label binarizer
        lb_path = output_dir / "label_binarizer.pkl"
        joblib.dump(self.dm.binarizer, lb_path)
        print(f"  ✓ Label binarizer: {lb_path}")

        # Save labels
        labels_path = output_dir / "labels.npy"
        np.save(labels_path, np.asarray(self.dm.labels_for_splitter, dtype=object))
        print(f"  ✓ Labels: {labels_path}")

        # Save taxonomy path (for reference)
        if "taxonomy_path" in self.config:
            self.metadata["taxonomy_path"] = str(self.config["taxonomy_path"])

        # Save metadata
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"  ✓ Metadata: {metadata_path}")

        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Pre-extract and cache BERT or ELECTRA embeddings"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wos",
        choices=["wos", "bugs", "amazon", "bgc"],
        help="Dataset name"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bert",
        choices=["bert", "electra", "codebert", "bge", "sbert", "scibert"],
        help="Model type to use for embedding extraction"
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-uncased",
        help="BERT model name (used when --model-type=bert)"
    )
    parser.add_argument(
        "--bert-strategy",
        type=str,
        default=None,
        choices=["mean_pool", "concat_cls", "mean_last", "concat_mean_last", "cls", "pool_cls"],
        help="Override the BERT pooling strategy from the YAML config. "
             "mean_pool is an alias for mean_last (final layer, excl. [CLS] and [SEP]). "
             "Default: use the value in the dataset's YAML config."
    )
    parser.add_argument(
        "--electra-model",
        type=str,
        default="google/electra-base-discriminator",
        help="ELECTRA model name (used when --model-type=electra)"
    )
    parser.add_argument(
        "--codebert-model",
        type=str,
        default="microsoft/codebert-base",
        help="CodeBERT model name (used when --model-type=codebert)"
    )
    parser.add_argument(
        "--codebert-strategy",
        type=str,
        default="mean_pool",
        choices=["mean_pool", "cls"],
        help="Pooling strategy for CodeBERT: mean_pool (final layer, excl. CLS+padding) or cls (CLS token from a specific layer)"
    )
    parser.add_argument(
        "--codebert-cls-layer",
        type=int,
        default=-1,
        help="Which layer to extract the CLS token from when --codebert-strategy=cls (-1 = last layer, -7 = 6th from end, etc.)"
    )
    parser.add_argument(
        "--bge-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="BGE model name (used when --model-type=bge)"
    )
    parser.add_argument(
        "--sbert-model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence-Transformers model name (used when --model-type=sbert)"
    )
    parser.add_argument(
        "--scibert-model",
        type=str,
        default="allenai/scibert_scivocab_uncased",
        help="SciBERT model name (used when --model-type=scibert)"
    )
    parser.add_argument(
        "--scibert-strategy",
        type=str,
        default=None,
        choices=["mean_pool", "concat_cls", "mean_last", "concat_mean_last", "cls", "pool_cls"],
        help="Override the SciBERT pooling strategy from the YAML config. "
             "mean_pool is an alias for mean_last (final layer, excl. [CLS] and [SEP]). "
             "Default: use the value in the dataset's YAML config."
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
        help="Batch size for embedding extraction"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for cached embeddings (default: ./embeddings_cache_{model_type})"
    )

    args = parser.parse_args()

    # Extract embeddings
    extractor = EmbeddingExtractor(
        dataset_name=args.dataset,
        bert_model=args.bert_model,
        bert_strategy=args.bert_strategy,
        device=args.device,
        model_type=args.model_type,
        electra_model=args.electra_model,
        codebert_model=args.codebert_model,
        codebert_strategy=args.codebert_strategy,
        codebert_cls_layer=args.codebert_cls_layer,
        bge_model=args.bge_model,
        sbert_model=args.sbert_model,
        scibert_model=args.scibert_model,
        scibert_strategy=args.scibert_strategy,
    )

    # Default dir includes strategy so different strategies don't overwrite each other.
    # Use script dir (not CWD) because os.chdir(repo_root) has already run by this point.
    if args.output_dir is None:
        script_dir = Path(__file__).resolve().parent
        args.output_dir = str(
            script_dir / "precomputed_data"
            / f"embeddings_cache_{args.model_type}_{extractor.effective_strategy}"
        )

    embeddings = extractor.extract_all_embeddings(batch_size=args.batch_size)
    splits = extractor.create_cv_splits(
        embeddings
    )

    # Save cache
    output_path = extractor.save_cache(
        Path(args.output_dir),
        embeddings,
        splits
    )

    print(f"\n{'='*60}")
    print(f"SUCCESS: Embeddings cached to {output_path}")
    print(f"{'='*60}")
    print(f"\nTo load in your code:")
    print(f"  import numpy as np")
    print(f"  import joblib")
    print(f"  embeddings = np.load('{output_path}/embeddings.npy')")
    print(f"  train_idx = np.load('{output_path}/train_indices_fold_0.npy')")
    print(f"  test_idx = np.load('{output_path}/test_indices_fold_0.npy')")
    print(f"  label_binarizer = joblib.load('{output_path}/label_binarizer.pkl')")
    print(f"\nRaw embeddings shape: {embeddings.shape}")
    print(f"Apply PCA in downstream tasks if needed (e.g., in train_on_cached_embeddings.py)")


if __name__ == "__main__":
    main()
