"""Interface check for the detection pipeline.

This script is an interface sanity check, not a miniature replication of
the paper's detection evaluation. The paper's results use 19,001 packages
with 10-fold stratified cross-validation at the package level, which
requires the full corpus.

What this script does:
    1. Turns every shipped PA log into a token document via
       ``model.pa_features.pa_execute_to_document``.
    2. Prints a per-document token count so you can confirm the feature
       extractor consumes both malicious and benign logs without error.
    3. Fits a TF-IDF logistic-regression classifier on the bundled
       corpus (3 malicious case packages x 2 variants + 30 benign logs)
       and writes per-package probabilities to
       ``data/model_output/tfidf_lr_predictions.csv`` so you can confirm
       the end-to-end training + prediction path works.

The classifier hyperparameters here are small-corpus-friendly
(``min_df=1``, reduced SVD components). For paper-accurate settings
(``min_df=20``, 200 SVD components, 5-classifier sweep), use
``model.tfidf_classifier.train_and_eval`` with the full corpus.

Run:
    python model/run_detection_example.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.pa_features import pa_execute_to_document

PA_LOGS = ROOT / "data" / "pa_logs"
BENIGN_LOGS = ROOT / "data" / "pa_logs_benign"
OUT_DIR = ROOT / "data" / "model_output"

CASES = ["safepackage", "equests", "r4quests"]


def load_doc(path: Path) -> str:
    return pa_execute_to_document(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    mal_rows, ben_rows = [], []
    for name in CASES:
        for variant in ("origin", "ladar"):
            fp = PA_LOGS / f"{name}_{variant}_execute.json"
            if fp.is_file():
                mal_rows.append((f"{name}__{variant}", load_doc(fp), 1))
    for fp in sorted(BENIGN_LOGS.glob("*.json")):
        ben_rows.append((fp.stem, load_doc(fp), 0))
    if not ben_rows:
        raise SystemExit("No benign PA logs under data/pa_logs_benign/.")

    print(f"[feature-extraction] {len(mal_rows)} malicious documents")
    for pid, doc, _ in mal_rows:
        print(f"  {pid:32s}  tokens = {len(doc.split()):5d}")
    print(f"[feature-extraction] {len(ben_rows)} benign documents"
          f"  (min/median/max tokens = "
          f"{min(len(d.split()) for _, d, _ in ben_rows)}/"
          f"{sorted(len(d.split()) for _, d, _ in ben_rows)[len(ben_rows)//2]}/"
          f"{max(len(d.split()) for _, d, _ in ben_rows)})")

    # Fit the classifier on everything; emit per-package probabilities.
    rows = mal_rows + ben_rows
    docs = [d for _, d, _ in rows]
    labels = [y for _, _, y in rows]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\S+")
    X = vec.fit_transform(docs)
    n_comp = max(2, min(64, X.shape[1] - 1, X.shape[0] - 1))
    Xs = StandardScaler().fit_transform(
        TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X)
    )
    clf = LogisticRegression(C=1.0, class_weight="balanced",
                             max_iter=1000, random_state=42)
    clf.fit(Xs, labels)
    probs = clf.predict_proba(Xs)[:, 1]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "tfidf_lr_predictions.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "y_true", "p_malicious"])
        for (pid, _, y), prob in zip(rows, probs):
            w.writerow([pid, y, f"{prob:.4f}"])
    print(f"[classifier] wrote {csv_path}")
    summary = (
        f"n_documents = {len(rows)}  "
        f"(malicious = {len(mal_rows)}, benign = {len(ben_rows)})\n"
        f"vocabulary = {X.shape[1]} tokens, SVD components = {n_comp}\n"
    )
    (OUT_DIR / "tfidf_lr_metrics.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")


if __name__ == "__main__":
    main()
