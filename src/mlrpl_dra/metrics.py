from __future__ import annotations

import numpy as np


def confusion_counts(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    c = confusion_counts(y_true, y_prob, threshold)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        **{k: float(v) for k, v in c.items()},
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "balanced_accuracy") -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        scores = binary_metrics(y_true, y_prob, float(threshold))
        if metric not in scores:
            raise ValueError(f"Unknown threshold metric: {metric}")
        score = scores[metric]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score
