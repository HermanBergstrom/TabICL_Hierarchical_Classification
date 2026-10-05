"""
Analyze token length distributions across HTC datasets.

Shows what fraction of documents get truncated at 512 tokens and how much
content is lost — useful for deciding whether --disable-max-length is worth
the VRAM cost for long-context models like Qwen3-Embedding.

Usage:
    python analyze_token_lengths.py
    python analyze_token_lengths.py --datasets wos bugs amazon bgc
    python analyze_token_lengths.py --tokenizer /model-weights/Qwen3-Embedding-8B
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

repo_root = Path(__file__).parent / "dsi-nlp-publib" / "htc-survey-24"
sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

from src.dataset_tools.dataset_manager import DatasetManager
from src.dataset_tools import get_bgc_split_jsonl
from src.utils.generic_functions import load_yaml
from src.utils.text_utilities.preprocess import text_cleanup

CACHE_DIR = "/scratch/hermanb/.cache/huggingface/transformers"


def load_texts(dataset_name: str) -> list[str]:
    config_name = "amz" if dataset_name == "amazon" else dataset_name
    config_path = Path("config") / "BERT" / f"bert_{config_name}.yml"
    config = load_yaml(config_path)

    dm = DatasetManager(dataset_name, config)

    if dataset_name == "bgc":
        train_docs = get_bgc_split_jsonl("train")
        val_docs   = get_bgc_split_jsonl("val")
        test_docs  = get_bgc_split_jsonl("test")
        all_docs   = train_docs + val_docs + test_docs
        texts = [doc["text"] for doc in all_docs]
    else:
        texts = dm.data

    return text_cleanup(texts, remove_garbage=config.get("REMOVE_GARBAGE_TEXT", False))


def analyze(texts: list[str], tokenizer, thresholds=(128, 256, 512, 1024, 2048, 4096, 8192)):
    lengths = []
    batch_size = 256
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        ids = tokenizer(batch, add_special_tokens=True, truncation=False)["input_ids"]
        lengths.extend(len(x) for x in ids)

    lengths = np.array(lengths)
    n = len(lengths)

    print(f"  N = {n:,}")
    print(f"  min / mean / median / max = "
          f"{lengths.min():,} / {lengths.mean():.0f} / {np.median(lengths):.0f} / {lengths.max():,}")
    print(f"  Percentiles:")
    for p in (75, 90, 95, 99, 99.9):
        print(f"    p{p:5.1f}: {np.percentile(lengths, p):8.0f} tokens")

    print(f"  Truncation impact:")
    for t in thresholds:
        truncated = (lengths > t).sum()
        lost_tokens = np.maximum(lengths - t, 0).sum()
        total_tokens = lengths.sum()
        print(f"    >{t:5d} tokens: {truncated:6,} docs ({100*truncated/n:.1f}%)  "
              f"| tokens lost: {lost_tokens:,} ({100*lost_tokens/total_tokens:.1f}% of corpus)")

    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+",
        default=["wos", "bugs", "amazon", "bgc"],
        choices=["wos", "bugs", "amazon", "bgc"],
    )
    parser.add_argument(
        "--tokenizer",
        default="/model-weights/Qwen3-Embedding-8B",
        help="Tokenizer to use (default: Qwen3-Embedding-8B). "
             "Use bert-base-uncased for a BERT comparison.",
    )
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=CACHE_DIR)
    print()

    for dataset in args.datasets:
        print(f"{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
        print("  Loading and cleaning texts...")
        texts = load_texts(dataset)
        print(f"  Tokenizing {len(texts):,} documents...")
        analyze(texts, tokenizer)
        print()


if __name__ == "__main__":
    main()
