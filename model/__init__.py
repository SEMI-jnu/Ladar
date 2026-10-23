"""LADAR downstream detectors.

Two families are provided, aligned with Section 5.3 of the paper:
- ``tfidf_classifier``: TF-IDF + truncated SVD + one of
  {LR, KNN, SVM, RF, XGB}
- ``nebula``: a PA-adapted Nebula transformer (schema mapping in Table 2)

Both use per-package PA execution logs and binary labels. Classical models
use pa_execute_to_document; Nebula uses pa_execute_to_nebula_document.
"""
from . import tfidf_classifier, pa_features

__all__ = ["tfidf_classifier", "pa_features"]
