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
    config = load_yaml(Path("config/BERT/bert_bgc.yml"))
    out_folder = Path("dumps/BERT") / f"bert_multilabel_BGC_concat_cls_3/run_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_training(config=config, dataset="bgc", model_class=BERTForClassification, out_folder=out_folder)
