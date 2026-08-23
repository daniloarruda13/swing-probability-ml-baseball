"""Leakage-resistant baseline pipeline and probability metrics."""

from __future__ import annotations


def make_baseline_pipeline(*, include_player_ids: bool = False, random_state: int = 42):
    """Build a train-fitted preprocessing and logistic-regression baseline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PLAYER_ID_FEATURES

    categorical = list(CATEGORICAL_FEATURES)
    if include_player_ids:
        categorical = [*PLAYER_ID_FEATURES, *categorical]
    numeric = list(NUMERIC_FEATURES)
    preprocessor = ColumnTransformer(
        transformers=(
            (
                "numeric",
                Pipeline(
                    steps=(
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    )
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    steps=(
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    )
                ),
                categorical,
            ),
        )
    )
    classifier = LogisticRegression(max_iter=1000, random_state=random_state)
    return Pipeline(steps=(("preprocess", preprocessor), ("classifier", classifier)))


def evaluate_probabilities(target, probabilities, *, threshold: float = 0.5) -> dict[str, float]:
    """Calculate accuracy, Brier score, and ROC AUC after strict validation."""
    import numpy as np
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

    target_array = np.asarray(target)
    probability_array = np.asarray(probabilities, dtype=float)
    if target_array.ndim != 1 or probability_array.ndim != 1:
        raise ValueError("target and probabilities must be one-dimensional")
    if target_array.shape[0] != probability_array.shape[0] or target_array.size == 0:
        raise ValueError("target and probabilities must have the same nonzero length")
    if not np.isin(target_array, (0, 1)).all():
        raise ValueError("target must contain only binary 0/1 values")
    valid_probabilities = (
        np.isfinite(probability_array).all()
        and ((0 <= probability_array) & (probability_array <= 1)).all()
    )
    if not valid_probabilities:
        raise ValueError("probabilities must be finite values between 0 and 1")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1")
    if np.unique(target_array).size < 2:
        raise ValueError("ROC AUC requires both target classes")

    predictions = (probability_array >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(target_array, predictions)),
        "brier_score": float(brier_score_loss(target_array, probability_array)),
        "roc_auc": float(roc_auc_score(target_array, probability_array)),
    }
