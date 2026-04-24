"""TF-IDF based malicious-package detection pipeline (Section 5.3.1).

Hyperparameters mirror the paper:
    TF-IDF:  unigram + bigram, min_df=20, max_features=120_000
    SVD:     200 components (standardized)
    LR:      C=1.0, class-balanced
    KNN:     k=25, distance-weighted
    SVM:     RBF kernel, C=2.0, class-balanced
    RF:      200 estimators
    XGB:     300 estimators, learning_rate=0.1

The fit/predict interface operates on lists of PA-execute "documents"
produced by :mod:`model.pa_features`.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


CLASSIFIERS = {
    "LR":  lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=25, weights="distance"),
    "SVM": lambda: SVC(C=2.0, kernel="rbf", class_weight="balanced", probability=True, random_state=42),
    "RF":  lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}
if _HAS_XGB:
    CLASSIFIERS["XGB"] = lambda: XGBClassifier(
        n_estimators=300, learning_rate=0.1, random_state=42,
        eval_metric="logloss", use_label_encoder=False, n_jobs=-1,
    )


def build_feature_matrix(
    train_docs: Sequence[str],
    test_docs: Sequence[str],
    *,
    n_components: int = 200,
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Fit TF-IDF+SVD on train, project both sets. Paper settings."""
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=20,
        max_features=120_000,
        token_pattern=r"(?u)\S+",
    )
    X_tr = vec.fit_transform(train_docs)
    X_te = vec.transform(test_docs)
    svd = TruncatedSVD(n_components=min(n_components, X_tr.shape[1] - 1), random_state=42)
    X_tr_s = svd.fit_transform(X_tr)
    X_te_s = svd.transform(X_te)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_s)
    X_te_s = scaler.transform(X_te_s)
    return X_tr_s, X_te_s, vec


def train_and_eval(
    train_docs: Sequence[str],
    train_labels: Sequence[int],
    test_docs: Sequence[str],
    test_labels: Sequence[int],
    *,
    classifiers: Sequence[str] = ("LR", "KNN", "SVM", "RF", "XGB"),
) -> List[dict]:
    """Fit each classifier on train, evaluate on test. Returns one row
    per classifier.
    """
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    X_tr, X_te, _ = build_feature_matrix(train_docs, test_docs)

    results = []
    for name in classifiers:
        if name not in CLASSIFIERS:
            continue
        clf = CLASSIFIERS[name]()
        clf.fit(X_tr, train_labels)
        y_pred = clf.predict(X_te)
        results.append({
            "model": name,
            "precision": float(precision_score(test_labels, y_pred, zero_division=0)),
            "recall":    float(recall_score(test_labels, y_pred, zero_division=0)),
            "f1":        float(f1_score(test_labels, y_pred, zero_division=0)),
            "accuracy":  float(accuracy_score(test_labels, y_pred)),
            "tp":        int(((y_pred == 1) & (test_labels == 1)).sum()),
            "fp":        int(((y_pred == 1) & (test_labels == 0)).sum()),
            "fn":        int(((y_pred == 0) & (test_labels == 1)).sum()),
            "tn":        int(((y_pred == 0) & (test_labels == 0)).sum()),
        })
    return results
