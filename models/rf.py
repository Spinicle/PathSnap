
"""Random Forest evaluation using Yadu's generated training feature dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from common.config import REPO_ROOT
from features.build_features import FEATURE_COLS
from models.common_eval import evaluate_model


def find_feature_dataset() -> Path:
    """Resolve Yadu's generated train dataset without fabricating a fallback."""
    configured = os.environ.get("PATHSNAP_FEATURES")
    if configured:
        path = Path(configured)
    else:
        path = REPO_ROOT / "cache" / "features_train.parquet"

    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Yadu's real generated feature dataset was not found: {path}. "
            "Set PATHSNAP_FEATURES to the generated features parquet file "
            "or run Yadu's build_features CLI to create cache/features_train.parquet."
        )

    return path


def load_real_training_data(
    dataset_path: Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load the real feature table and verify Yadu's exact schema."""
    path = dataset_path or find_feature_dataset()
    data = pd.read_parquet(path)

    required = {"trip_id", "obs_idx", "arc_id", "label", *FEATURE_COLS}
    missing = sorted(required.difference(data.columns))

    if missing:
        raise ValueError(
            f"Generated feature dataset {path} is missing Yadu's required columns: {missing}"
        )

    if data["label"].isna().any():
        raise ValueError("The target column 'label' contains missing values.")

    labels = set(data["label"].unique().tolist())

    if not labels.issubset({-1, 0, 1}):
        raise ValueError("The target column 'label' must contain only -1, 0, or 1.")

    binary_labels = data.loc[data["label"] != -1, "label"]

    if set(binary_labels.unique().tolist()) != {0, 1}:
        raise ValueError(
            "Cross-validation requires both label 0 and label 1 in the training data."
        )

    data = data.loc[data["label"] != -1].copy()

    if data.empty:
        raise ValueError("No labeled candidate rows remain after excluding label=-1.")

    return data, list(FEATURE_COLS)


def run_random_forest(
    dataset_path: Path | None = None,
    n_splits: int = 5,
):
    """Evaluate a resource-conscious Random Forest using grouped CV."""

    data, feature_columns = load_real_training_data(dataset_path)

    print(f"Loaded dataset: {len(data):,} candidate rows")
    print(f"Training trips: {data['trip_id'].nunique()}")
    print(f"Features: {len(feature_columns)}")
    print("Starting 5-fold GroupKFold Random Forest evaluation...")

    estimator = RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=2,
        random_state=42,
        verbose=1,
    )

    return evaluate_model(
        estimator=estimator,
        X=data[feature_columns],
        y=data["label"],
        groups=data["trip_id"],
        observation_ids=data["obs_idx"],
        model_name="random_forest",
        n_splits=n_splits,
        random_state=42,
        candidate_ids=data["arc_id"],
        results_dir=REPO_ROOT / "results" / "runs",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Random Forest on Yadu's real generated feature dataset."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional path to cache/features_train.parquet.",
    )

    parser.add_argument("--n-splits", type=int, default=5)

    args = parser.parse_args()

    results = run_random_forest(args.dataset, args.n_splits)

    print("\nRandom Forest evaluation results:")
    print(results.to_string(index=False))
