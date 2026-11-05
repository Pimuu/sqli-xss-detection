"""Metric computation for 3-class and derived binary (malicious vs benign) views."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from . import config

LABELS = config.LABELS


def multiclass_metrics(y_true, y_pred) -> dict:
    """Accuracy, macro and per-class precision / recall / F1."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                                 average="macro", zero_division=0)
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "macro_precision": p, "macro_recall": r, "macro_f1": f}
    pc, rc, fc, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                                    average=None, zero_division=0)
    for i, lab in enumerate(LABELS):
        out[f"{lab}_precision"], out[f"{lab}_recall"], out[f"{lab}_f1"] = pc[i], rc[i], fc[i]
    return out


def binary_metrics(y_true, y_pred, score=None, hard_mask=None) -> dict:
    """Collapse SQLi+XSS into 'malicious' and report detection metrics.

    * `bin_fpr`            - share of benign inputs flagged as malicious
    * `hard_benign_fpr`    - the same, restricted to the hard-benign subset
    * `roc_auc`            - threshold-free ranking quality of `score`
    """
    t = np.asarray(y_true) != "benign"
    p = np.asarray(y_pred) != "benign"
    tp, fp = np.sum(t & p), np.sum(~t & p)
    fn, tn = np.sum(t & ~p), np.sum(~t & ~p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "bin_precision": prec, "bin_recall": rec,
        "bin_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "bin_fpr": fp / (fp + tn) if fp + tn else 0.0,
    }
    if score is not None and t.any() and (~t).any():
        out["roc_auc"] = roc_auc_score(t, score)
    if hard_mask is not None:
        hm = np.asarray(hard_mask, dtype=bool) & ~t
        out["hard_benign_fpr"] = float(p[hm].mean()) if hm.any() else np.nan
        em = ~np.asarray(hard_mask, dtype=bool) & ~t
        out["easy_benign_fpr"] = float(p[em].mean()) if em.any() else np.nan
    return out


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=LABELS)


def recall_by_group(df: pd.DataFrame, y_pred, group_col: str) -> pd.DataFrame:
    """Per-group recall (share of rows whose prediction equals the true label)."""
    d = df.assign(correct=(np.asarray(y_pred) == df["label"].to_numpy()),
                  detected=(np.asarray(y_pred) != "benign"))
    g = d.groupby(group_col).agg(n=("correct", "size"), exact_recall=("correct", "mean"),
                                 detection_rate=("detected", "mean"))
    return g.reset_index()
