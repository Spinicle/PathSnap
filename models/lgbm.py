"""LightGBM candidate classifier (Owner: Yajat).

Run the shared GroupKFold and student-test evaluation with:
    python -m models.lgbm
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from features.build_features import FEATURE_COLS


def make_model(seed: int = 42, **params) -> LGBMClassifier:
    config = {
        "objective": "binary",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
    }
    config.update(params)
    return LGBMClassifier(**config)


def fit_model(
    features: pd.DataFrame,
    feature_cols: list[str] | None = None,
    seed: int = 42,
    **params,
) -> LGBMClassifier:
    """Fit on labeled candidate rows; trip/grouping is handled by shared CV code."""
    cols = FEATURE_COLS if feature_cols is None else feature_cols
    missing = [col for col in cols if col not in features.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")
    if "label" not in features.columns:
        raise ValueError("Training features must include a label column")

    labels = features["label"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must contain only 0 and 1")
    if np.unique(labels).size != 2:
        raise ValueError("LightGBM training requires both label classes")

    x = features[cols].to_numpy(dtype=np.float32)
    if np.isinf(x).any():
        raise ValueError("Model features cannot contain positive or negative infinity")

    model = make_model(seed=seed, **params)
    model.fit(x, labels.astype(np.int8))
    return model


def predict_proba(
    model: LGBMClassifier,
    features: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Return P(label=1) for each candidate row, matching the shared model contract."""
    cols = FEATURE_COLS if feature_cols is None else feature_cols
    missing = [col for col in cols if col not in features.columns]
    if missing:
        raise ValueError(f"Missing model feature columns: {missing}")

    classes = np.asarray(model.classes_)
    positive_class = np.flatnonzero(classes == 1)
    if positive_class.size != 1:
        raise ValueError("Fitted model does not expose a unique positive class")
    return model.predict_proba(features[cols].to_numpy(dtype=np.float32))[:, positive_class[0]]


def run() -> LGBMClassifier:
    """Run the same grouped-CV and student-test protocol as the baselines."""
    from models.logreg import run_baseline

    return run_baseline("lgbm", make_model())


if __name__ == "__main__":
    run()
