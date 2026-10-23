"""Classical detectors using the Table 8 preprocessing and validation protocol.
Pass aligned ORIGIN documents even when training on repaired logs.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .pa_features import keep_signal_tokens

SEED = 42
MODELS = ("LR", "KNN", "SVM", "RF", "XGB")


def make_classifier(name: str, fold: int = 0):
    seed = SEED + fold
    if name == "LR":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, n_jobs=4, random_state=seed)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    if name == "SVM":
        base = SVC(C=2.0, kernel="rbf", class_weight="balanced", probability=False, random_state=seed)
        return CalibratedClassifierCV(base, method="isotonic", cv=3)
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, max_depth=20, class_weight="balanced_subsample", n_jobs=4, random_state=seed)
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             eval_metric="logloss", n_jobs=4, random_state=seed)
    raise ValueError(f"Unknown classifier: {name}")


def fit_origin_vectorizer(origin_fit_docs: Sequence[str]) -> TfidfVectorizer:
    vec = TfidfVectorizer(analyzer="word", tokenizer=keep_signal_tokens,
                          preprocessor=None, lowercase=False, token_pattern=None,
                          ngram_range=(1, 2), min_df=10, max_df=0.98,
                          max_features=360_000, dtype=np.float32)
    return vec.fit(origin_fit_docs)


def select_threshold_f1(labels, probabilities) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    scores = 2 * precision * recall / (precision + recall + 1e-9)
    best = int(np.nanargmax(scores))
    return float(thresholds[best]) if best < len(thresholds) else 0.5


def metrics_at_threshold(labels, probabilities, threshold: float) -> dict:
    y = np.asarray(labels)
    pred = np.asarray(probabilities) >= threshold
    return dict(precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)),
                f1=float(f1_score(y, pred, zero_division=0)),
                accuracy=float(accuracy_score(y, pred)),
                tp=int(((y == 1) & pred).sum()), fp=int(((y == 0) & pred).sum()),
                fn=int(((y == 1) & ~pred).sum()), tn=int(((y == 0) & ~pred).sum()))


def train_and_eval(train_docs: Sequence[str], train_labels: Sequence[int],
                   test_docs: Sequence[str], test_labels: Sequence[int], *,
                   origin_train_docs: Sequence[str], classifiers: Sequence[str] = MODELS,
                   fold: int = 0) -> list[dict]:
    """One outer fold; reserve 15% of its training pool for validation.

    ORIGIN and scenario documents must have identical package order.
    The same labels, ORIGIN documents and fold index produce identical
    inner splits and vocabulary across calls for different scenarios.
    """
    y = np.asarray(train_labels)
    if len(train_docs) != len(y) or len(origin_train_docs) != len(y):
        raise ValueError("Training labels, scenario and ORIGIN documents must be aligned.")
    if len(test_docs) != len(test_labels):
        raise ValueError("Test documents and labels must be aligned.")
    fit, val = train_test_split(np.arange(len(y)), test_size=0.15,
                                stratify=y, random_state=SEED + fold)
    vec = fit_origin_vectorizer([origin_train_docs[i] for i in fit])
    x_fit = vec.transform([train_docs[i] for i in fit])
    x_val = vec.transform([train_docs[i] for i in val])
    x_test = vec.transform(test_docs)
    svd = TruncatedSVD(n_components=min(200, max(2, x_fit.shape[1] - 1)), random_state=SEED)
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(svd.fit_transform(x_fit))
    x_val = scaler.transform(svd.transform(x_val))
    x_test = scaler.transform(svd.transform(x_test))
    results = []
    for name in classifiers:
        clf = make_classifier(name, fold)
        clf.fit(x_fit, y[fit])
        threshold = select_threshold_f1(y[val], clf.predict_proba(x_val)[:, 1])
        probs = clf.predict_proba(x_test)[:, 1]
        results.append(dict(model=name, threshold=threshold,
                            vocabulary_size=len(vec.vocabulary_),
                            **metrics_at_threshold(test_labels, probs, threshold)))
    return results


def evaluate_scenarios(scenario_docs: dict[str, Sequence[str]], labels: Sequence[int],
                       *, classifiers: Sequence[str] = MODELS) -> list[dict]:
    """Ten paired package-level folds on already selected, aligned logs.

    Supply one document per package and scenario, retaining empty ORIGIN
    documents. Apply the log-selection and population rules of Section 5.4
    before calling this function.
    """
    if "ORIGIN" not in scenario_docs:
        raise ValueError("ORIGIN documents are required to fix the vocabulary.")
    y = np.asarray(labels)
    if any(len(docs) != len(y) for docs in scenario_docs.values()):
        raise ValueError("All scenarios must have identical package order and length.")
    rows = []
    splitter = StratifiedKFold(10, shuffle=True, random_state=SEED)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
        origin = [scenario_docs["ORIGIN"][i] for i in train]
        for scenario, docs in scenario_docs.items():
            results = train_and_eval([docs[i] for i in train], y[train],
                                     [docs[i] for i in test], y[test],
                                     origin_train_docs=origin, classifiers=classifiers, fold=fold)
            rows.extend(dict(scenario=scenario, fold=fold, **r) for r in results)
    return rows
