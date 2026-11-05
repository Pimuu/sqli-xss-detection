"""Model zoo: every model is a single scikit-learn Pipeline (raw string -> label).

Wrapping feature extraction inside the pipeline guarantees that vocabulary,
IDF weights and scaling are learned on training folds only (no leakage), and
that a fitted model can be applied directly to raw request strings.

| key          | features                     | classifier                      |
|--------------|------------------------------|---------------------------------|
| tfidf_nb     | char 1-4 gram TF-IDF         | Multinomial Naive Bayes (baseline)
| tfidf_lr     | char 1-4 gram TF-IDF         | Logistic Regression (multinomial)
| tfidf_svm    | char 1-4 gram TF-IDF         | Linear SVM (one-vs-rest)
| hand_rf      | 28 hand-crafted features     | Random Forest (300 trees)
| hybrid_lr    | TF-IDF + scaled hand-crafted | Logistic Regression
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from . import config
from .features import HandcraftedFeatures, build_tfidf

MODEL_NAMES = {
    "tfidf_nb": "TF-IDF + Naive Bayes",
    "tfidf_lr": "TF-IDF + LogReg",
    "tfidf_svm": "TF-IDF + Linear SVM",
    "hand_rf": "Hand-crafted + Random Forest",
    "hybrid_lr": "Hybrid + LogReg",
}


def _log1p_counts(X):
    """Compress heavy-tailed counts (lengths, keyword counts) before scaling."""
    return np.log1p(np.abs(X)) * np.sign(X)


def build_model(key: str, canonical: bool = True, seed: int = config.SEED) -> Pipeline:
    """Return an unfitted Pipeline for model `key`."""
    if key == "tfidf_nb":
        return Pipeline([("tfidf", build_tfidf(canonical)),
                         ("clf", MultinomialNB(alpha=0.1))])
    if key == "tfidf_lr":
        return Pipeline([("tfidf", build_tfidf(canonical)),
                         ("clf", LogisticRegression(C=10.0, max_iter=3000, random_state=seed))])
    if key == "tfidf_svm":
        return Pipeline([("tfidf", build_tfidf(canonical)),
                         ("clf", LinearSVC(C=1.0, random_state=seed))])
    if key == "hand_rf":
        return Pipeline([("hand", HandcraftedFeatures(canonical)),
                         ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
                                                        n_jobs=-1, random_state=seed))])
    if key == "hybrid_lr":
        hand = Pipeline([("hand", HandcraftedFeatures(canonical)),
                         ("log", FunctionTransformer(_log1p_counts)),
                         ("scale", StandardScaler())])
        union = FeatureUnion([("tfidf", build_tfidf(canonical)), ("hand", hand)])
        return Pipeline([("features", union),
                         ("clf", LogisticRegression(C=10.0, max_iter=3000, random_state=seed))])
    raise KeyError(f"unknown model '{key}', choose from {list(MODEL_NAMES)}")


def malicious_score(model: Pipeline, X) -> np.ndarray:
    """Continuous 'is malicious' score for binary ROC/AUC analysis.

    Probabilistic models: 1 - P(benign). Margin models (LinearSVC):
    max(margin_sqli, margin_xss) - margin_benign.
    """
    classes = list(model.classes_)
    b = classes.index("benign")
    if hasattr(model, "predict_proba"):
        return 1.0 - model.predict_proba(X)[:, b]
    d = model.decision_function(X)
    others = [i for i in range(len(classes)) if i != b]
    return d[:, others].max(axis=1) - d[:, b]
