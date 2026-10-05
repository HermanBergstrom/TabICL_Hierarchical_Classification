"""
Standalone SVM baseline that mirrors the paper's TF-IDF + LinearSVC pipeline.

Uses the same DatasetManager / splitter / seeds as the rest of the pipeline so
results are directly comparable to TabICL runs.

Usage
-----
    python run_svm_baseline.py --dataset wos
    python run_svm_baseline.py --dataset amazon --max-features 50000
    python run_svm_baseline.py --dataset wos --grid-cv 3
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD

# ── repo root on sys.path ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR / "dsi-nlp-publib" / "htc-survey-24"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from src.dataset_tools.dataset_manager import DatasetManager
from src.models.SVM.preprocessing import deeptriage_list_processing
from src.utils.generic_functions import load_yaml


# ── helpers ───────────────────────────────────────────────────────────────────

def _config_path(dataset: str) -> Path:
    name = "amz" if dataset == "amazon" else dataset
    #return Path("config") / "BERT" / f"bert_{name}.yml"
    return Path("config") / "SVM" / f"svm_config_{name}_mc.yml"


def _preprocess(texts: List[str], config: Dict) -> List[str]:
    return deeptriage_list_processing(
        texts,
        remove_garbage=config.get("remove_garbage", False),
        stop_words_removal=config.get("stop_words_removal", False),
        join=True,
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division="warn"
    )
    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc, "macro_f1": f1, "macro_precision": prec, "macro_recall": rec}


# ── main logic ────────────────────────────────────────────────────────────────

def run(
    dataset: str,
    max_features: int,
    ngram_max: int,
    grid_params: Dict,
    grid_cv: int,
    chi2_features: int = None,
) -> None:
    config = load_yaml(_config_path(dataset))

    print(f"Loading dataset: {dataset}")
    dm = DatasetManager(dataset, config)

    print(f"Preprocessing {len(dm.data)} documents...")
    all_texts = _preprocess(dm.data, config)

    all_texts_arr = np.array(all_texts, dtype=object)
    all_labels    = dm.labels_for_splitter  # leaf label per sample for stratification
    # binarizer was fit on all labels in DatasetManager.__init__
    binarizer = dm.binarizer
    results: List[Dict] = []
    fold_idx = 0

    for train_idx, test_idx in dm.splitter.split(dm.data, all_labels):
        fold_idx += 1
        print(f"\n─── Fold {fold_idx} (train={len(train_idx)}, test={len(test_idx)}) ───")

        x_train = all_texts_arr[train_idx].tolist()
        x_test  = all_texts_arr[test_idx].tolist()

        # Binarize labels — multilabel: binarizer; multiclass: label array
        train_labels_raw = [all_labels[i] for i in train_idx]
        test_labels_raw  = [all_labels[i] for i in test_idx]

        # Re-fit binarizer on train split to avoid leakage (same classes, consistent ordering)
        if binarizer is not None:
            y_train = binarizer.transform([[lbl] for lbl in train_labels_raw])
            y_test  = binarizer.transform([[lbl] for lbl in test_labels_raw])
        else:
            # multiclass: integer-encode via the same class ordering seen globally
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder().fit(all_labels)
            y_train = le.transform(train_labels_raw)

            y_test  = le.transform(test_labels_raw)
        # TF-IDF — fit on train only; stop words are handled in preprocessing, not here
        vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, ngram_max),
        )
        X_train = vec.fit_transform(x_train)
        X_test  = vec.transform(x_test)

        if chi2_features is not None and chi2_features < X_train.shape[1]:
            if y_train.ndim > 1:
                # multilabel: average chi2 scores across labels, then pick top-k
                scores = np.zeros(X_train.shape[1])
                for j in range(y_train.shape[1]):
                    col_scores, _ = chi2(X_train, y_train[:, j])
                    scores += col_scores
                top_idx = np.argsort(scores)[-chi2_features:]
                X_train = X_train[:, top_idx]
                X_test  = X_test[:, top_idx]
            else:
                sel = SelectKBest(chi2, k=chi2_features)
                X_train = sel.fit_transform(X_train, y_train)
                print(sel.scores_)
                X_test  = sel.transform(X_test)
            print(f"  After chi2 selection: train={X_train.shape}, test={X_test.shape}")

        #TODO comment/uncomment
        #svd = TruncatedSVD(n_components=768, random_state=42)
        #X_train = svd.fit_transform(X_train)#.astype(np.float32)
        #X_test  = svd.transform(X_test)#.astype(np.float32)

        print(f"  Feature matrix: train={X_train.shape}, test={X_test.shape}")

        # Classifier — always GridSearch + retrain, matching the survey pipeline
        base_clf = OneVsRestClassifier(LinearSVC(), n_jobs=4)
        grid_clf = GridSearchCV(
            base_clf,
            grid_params,
            cv=grid_cv,
            scoring="f1_macro",
            verbose=1,
            n_jobs=4,
        )
        grid_clf.fit(X_train, y_train)
        print(f"  Best params: {grid_clf.best_params_}")
        clf = OneVsRestClassifier(
            LinearSVC(C=grid_clf.best_params_["estimator__C"],
                      max_iter=grid_clf.best_params_["estimator__max_iter"]),
            n_jobs=4,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        metrics = _metrics(y_test, y_pred)
        print(f"  macro-F1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}")
        results.append(metrics)

    # Summary
    df = pd.DataFrame(results)
    df.index = [f"fold_{i+1}" for i in range(len(results))]
    df.loc["mean"] = df.mean()
    df.loc["std"]  = df.iloc[:-1].std()

    print("\n══ Results ══")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    out_dir = SCRIPT_DIR / "out" / "results" / "SVM" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "svm_results.csv"
    df.to_csv(out_path)
    print(f"\nSaved to {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SVM TF-IDF baseline")
    parser.add_argument("--dataset", required=True,
                        choices=["wos", "bugs", "amazon", "rcv1", "blurb", "bgc"],
                        help="Dataset name")
    parser.add_argument("--max-features", type=int, default=30000,
                        help="TF-IDF vocabulary size (default: 30000)")
    parser.add_argument("--ngram-max", type=int, default=2,
                        help="Upper bound of n-gram range (default: 2 → unigrams+bigrams)")
    parser.add_argument("--grid-cv", type=int, default=2,
                        help="Inner CV folds for GridSearchCV (default: 2, matching survey)")
    parser.add_argument("--chi2-features", type=int, default=None,
                        help="Keep top-k features by chi2 score after TF-IDF (default: disabled)")
    args = parser.parse_args()

    # Matches svm_config_*_mc.yml SVM_GRID_PARAMS
    grid_params = {
        "estimator__C": [0.01, 0.1, 1, 5, 10],
        "estimator__max_iter": [200],
    }

    run(
        dataset=args.dataset,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
        grid_params=grid_params,
        grid_cv=args.grid_cv,
        chi2_features=args.chi2_features,
    )


if __name__ == "__main__":
    main()
