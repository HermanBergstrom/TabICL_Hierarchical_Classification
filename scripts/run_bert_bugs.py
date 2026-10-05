"""
BERT flat baseline — Bugs dataset.

Run from the htc-survey-24 directory on a GPU node:
    python run_bert_bugs.py

Allocate a GPU node first:
    srun --account="aip-rahulgk" --gres=gpu:l40s:1 --mem=32G -c 4 --time=06:00:00 --pty bash
"""
import datetime as dt
import torch
from pathlib import Path

from src.models.BERT_flat.bert.bert_classifier import BERTForClassification
from src.training_scripts.flat.bert import run_training
from src.utils.generic_functions import load_yaml

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = load_yaml(Path("config/BERT/bert_bugs.yml"))
    out_folder = (
        Path("dumps/BERT")
        / f"bert_multilabel_bugs_concat_cls_3/run_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    run_training(
        config=config,
        dataset="bugs",
        model_class=BERTForClassification,
        out_folder=out_folder,
    )
