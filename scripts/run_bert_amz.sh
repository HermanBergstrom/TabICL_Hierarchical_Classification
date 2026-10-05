#!/bin/bash
# Run BERT flat baseline on Amazon dataset.
# Execute inside a GPU node, then: bash run_bert_amz.sh
# Note: Amazon is large (500k samples). Allow ~8-12 hours for 6 folds.

set -e

ENV="/project/aip-rahulgk/hermanb/environments/aditya_tabicl"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

source "${ENV}/bin/activate"
cd "${REPO_DIR}"

echo "Working directory: $(pwd)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -c "
import datetime as dt
from pathlib import Path
from src.models.BERT_flat.bert.bert_classifier import BERTForClassification
from src.training_scripts.flat.bert import run_training
from src.utils.generic_functions import load_yaml

config = load_yaml(Path('config/BERT/bert_amz.yml'))
out_folder = Path('dumps/BERT') / f\"bert_multilabel_AMZ_concat_cls_3/run_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}\"
run_training(config=config, dataset='amazon', model_class=BERTForClassification, out_folder=out_folder)
"
