from __future__ import annotations

import numpy as np

from .common import LABELS, TERMINALS


def _prf(tp: int, predicted: int, gold: int) -> dict:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "support": int(gold), "predicted": int(predicted)}


def scores(confusion: np.ndarray) -> dict:
    confusion = np.asarray(confusion, dtype=np.int64)
    if confusion.shape != (len(LABELS), len(LABELS)):
        raise ValueError("Unexpected confusion matrix dimensions")
    per_class = {label: _prf(int(confusion[i, i]), int(confusion[:, i].sum()), int(confusion[i, :].sum())) for i, label in enumerate(LABELS)}
    positive_tp = int(np.diag(confusion)[1:].sum())
    micro = _prf(positive_tp, int(confusion[:, 1:].sum()), int(confusion[1:, :].sum()))
    indices = sorted(TERMINALS)
    boundary = _prf(int(confusion[np.ix_(indices, indices)].sum()), int(confusion[:, indices].sum()), int(confusion[indices, :].sum()))
    return {"characters": int(confusion.sum()), "accuracy_including_O": float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0, "punctuation_micro": micro, "punctuation_macro_f1": float(np.mean([per_class[label]["f1"] for label in LABELS[1:]])), "sentence_boundary": boundary, "per_class": per_class, "confusion_gold_rows_predicted_columns": confusion.tolist()}


def add_confusion(confusion: np.ndarray, gold, predicted) -> None:
    gold, predicted = np.asarray(gold), np.asarray(predicted)
    valid = gold != -100
    np.add.at(confusion, (gold[valid], predicted[valid]), 1)
