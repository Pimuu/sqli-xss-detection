"""Sliding-window inference: a defence against benign-padding (mimicry) evasion.

A whole-string TF-IDF vector is L2-normalised, so surrounding a short payload
with a few hundred characters of innocuous prose dilutes the weight of every
attack n-gram ("payload dilution"). `WindowedDetector` instead scores
overlapping windows of the canonicalised string and reports the *most
suspicious* window (max-pooling), so padding can no longer drown the payload.

It wraps an already-fitted pipeline - no retraining is required.
"""
from __future__ import annotations

import numpy as np

from .preprocessing import canonicalize


def split_windows(text: str, width: int, stride: int) -> list[str]:
    """Overlapping character windows of the canonical string (>=1 window)."""
    c = canonicalize(text)
    if len(c) <= width:
        return [c]
    starts = range(0, len(c) - width + stride, stride)
    return [c[i:i + width] for i in starts]


class WindowedDetector:
    """Max-pool a fitted 3-class pipeline over sliding windows."""

    def __init__(self, model, width: int = 48, stride: int = 16):
        self.model, self.width, self.stride = model, width, stride
        self.classes_ = np.asarray(model.classes_)
        self._b = list(self.classes_).index("benign")
        self._prob = hasattr(model, "predict_proba")

    def _window_scores(self, X):
        owner, segs = [], []
        for i, s in enumerate(X):
            w = split_windows(str(s), self.width, self.stride)
            segs.extend(w)
            owner.extend([i] * len(w))
        owner = np.asarray(owner)
        if self._prob:
            S = self.model.predict_proba(segs)
            mal = 1.0 - S[:, self._b]
        else:
            S = self.model.decision_function(segs)
            others = [k for k in range(S.shape[1]) if k != self._b]
            mal = S[:, others].max(axis=1) - S[:, self._b]
        # index of the most malicious window for every input string
        n = int(owner.max()) + 1 if len(owner) else 0
        best = np.full(n, -1)
        best_score = np.full(n, -np.inf)
        for k, (o, m) in enumerate(zip(owner, mal)):
            if m > best_score[o]:
                best_score[o], best[o] = m, k
        return S[best], best_score

    def malicious_score(self, X) -> np.ndarray:
        return self._window_scores(X)[1]

    def predict(self, X) -> np.ndarray:
        S, score = self._window_scores(X)
        threshold = 0.5 if self._prob else 0.0
        masked = S.copy()
        masked[:, self._b] = -np.inf                      # best attack class
        attack = self.classes_[np.argmax(masked, axis=1)]
        return np.where(score > threshold, attack, "benign")
