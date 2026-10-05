"""
Shared flat-metric helpers that match the htc-survey-24 baseline's label space.

The baseline (BERT flat / Match / etc.) fits its MultiLabelBinarizer on ALL labels
— both parent/internal nodes and leaf nodes — and computes accuracy (exact match)
and macro F1 on the resulting multi-hot matrices.

Both helpers here replicate this by extending every predicted/true label set with
its taxonomy ancestors before encoding, so the label space and averaging are
identical to the baseline's compute_metrics(threshold=-1).

Active-column filtering: labels absent from both Y_true and Y_pred in a given
split are excluded from the macro F1 average.  This matches sklearn 1.1.1
(the version used by htc-survey-24), which silently excluded such labels rather
than assigning them F1=0.
"""

from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


def _extend_labels(
    label_sets: List[List[str]],
    parent_map: dict,
) -> List[List[str]]:
    """Walk up parent_map to add ancestor nodes (excluding 'root') to each label set."""
    result = []
    for labels in label_sets:
        extended = set(labels)
        for lbl in labels:
            current = lbl
            while current in parent_map:
                current = parent_map[current]
                extended.add(current)
        extended.discard("root")
        result.append(list(extended))
    return result


def _binarize_and_score(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> dict:
    """Compute exact-match accuracy and active-column macro F1 on multi-hot matrices."""
    acc = float(accuracy_score(Y_true, Y_pred))
    active = (Y_true.sum(axis=0) > 0) | (Y_pred.sum(axis=0) > 0)
    _, _, f1, _ = precision_recall_fscore_support(
        Y_true[:, active], Y_pred[:, active], average="macro", zero_division=0
    )
    return {"accuracy": acc, "macro_f1": float(f1)}


def compute_flat_metrics_from_sets(
    y_true_sets: List[List[str]],
    y_pred_sets: List[List[str]],
    parent_map: dict,
    mlb_hier: MultiLabelBinarizer,
) -> dict:
    """
    Compute flat accuracy and macro F1 for multi-label predictions.

    Intended for the hierarchical model where each sample may have multiple
    predicted labels (BFS visited nodes).  Each label set is extended with
    taxonomy ancestors before multi-hot encoding.

    Args:
        y_true_sets:  List of true label sets, one list per sample.
        y_pred_sets:  List of predicted label sets, one list per sample.
        parent_map:   {child: parent} dict for the full taxonomy (including "root").
        mlb_hier:     MultiLabelBinarizer fitted on all taxonomy nodes (excl. root).

    Returns:
        dict with keys "accuracy" and "macro_f1".
    """
    Y_true = mlb_hier.transform(_extend_labels(y_true_sets, parent_map))
    Y_pred = mlb_hier.transform(_extend_labels(y_pred_sets, parent_map))
    return _binarize_and_score(Y_true, Y_pred)


def compute_flat_metrics(
    y_true_1d: np.ndarray,
    y_pred_1d: np.ndarray,
    parent_map: dict,
    mlb_hier: MultiLabelBinarizer,
) -> dict:
    """
    Compute flat accuracy and macro F1 for flat (single-label) predictions.

    Intended for flat classifiers (TabICL flat, MLP) that predict one leaf
    label per sample.  Each label is extended with taxonomy ancestors before
    multi-hot encoding.

    Args:
        y_true_1d:   1-D array of true leaf-label strings, one per sample.
        y_pred_1d:   1-D array of predicted leaf-label strings, one per sample.
        parent_map:  {child: parent} dict for the full taxonomy (including "root").
        mlb_hier:    MultiLabelBinarizer fitted on all taxonomy nodes (excl. root).

    Returns:
        dict with keys "accuracy" and "macro_f1".
    """
    return compute_flat_metrics_from_sets(
        [[lbl] for lbl in y_true_1d],
        [[lbl] for lbl in y_pred_1d],
        parent_map,
        mlb_hier,
    )
