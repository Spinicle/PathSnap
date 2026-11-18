from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold


_METRIC_COLUMNS = (
    "accuracy",
    "precision",
    "recall",
    "macro_f1",
    "log_loss",
    "top1_accuracy",
    "candidate_miss_rate",
    "training_time_seconds",
    "inference_time_seconds",
    "confusion_tn",
    "confusion_fp",
    "confusion_fn",
    "confusion_tp",
)


def _as_1d_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; received shape {array.shape}.")
    return array


def _validate_non_missing(values: np.ndarray, name: str) -> None:
    """Reject missing identifiers or labels rather than silently dropping rows."""
    try:
        has_missing = bool(pd.isna(pd.Series(values)).any())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain scalar, non-missing values.") from exc
    if has_missing:
        raise ValueError(f"{name} must not contain missing values.")


def _validate_hashable(values: np.ndarray, name: str) -> None:
    for value in values:
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain hashable scalar values.") from exc


def _contains_only_binary_labels(values: np.ndarray) -> bool:
    try:
        return set(values.tolist()).issubset({0, 1})
    except TypeError:
        return False


def _take_rows(values: Any, indices: np.ndarray) -> Any:
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return values[indices]


def _seed_unset_random_states(estimator: BaseEstimator, seed: Optional[int]) -> None:
    """Set unset sklearn random_state parameters on a fold-local estimator."""
    if seed is None:
        return
    params = estimator.get_params(deep=True)
    unset_seeds = {
        name: seed
        for name, value in params.items()
        if (name == "random_state" or name.endswith("__random_state")) and value is None
    }
    if unset_seeds:
        estimator.set_params(**unset_seeds)


def _positive_class_probability(estimator: BaseEstimator, raw: Any, row_count: int) -> np.ndarray:
    """Extract P(label=1), including folds whose training data has one class."""
    probabilities = np.asarray(raw, dtype=float)
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        raise ValueError("predict_proba is available but the fitted estimator has no classes_ attribute.")
    classes = np.asarray(classes)

    if probabilities.ndim != 2 or probabilities.shape[0] != row_count:
        raise ValueError(
            "predict_proba must return a two-dimensional array with one row per validation sample."
        )
    if probabilities.shape[1] != len(classes):
        raise ValueError("predict_proba columns do not match the fitted estimator's classes_.")
    if not np.isfinite(probabilities).all():
        raise ValueError("predict_proba returned non-finite probability values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("predict_proba returned values outside [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError("predict_proba rows must sum to 1.")
    if not set(classes.tolist()).issubset({0, 1}):
        raise ValueError("The estimator classes_ must use the binary labels 0 and 1.")

    positive_indices = np.flatnonzero(classes == 1)
    if len(positive_indices):
        positive_probability = probabilities[:, positive_indices[0]]
    elif len(classes) == 1 and classes[0] == 0:
        positive_probability = np.zeros(row_count, dtype=float)
    else:
        raise ValueError("The fitted estimator does not expose a probability for label 1.")

    return positive_probability


def _observation_top1_accuracy(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    observation_ids: np.ndarray,
) -> float:
    """Top-1 over observations that have candidate rows; absent observations are not counted."""
    selected: dict[tuple[Any, Any], int] = {}
    for row_index, (trip_id, observation_id) in enumerate(zip(groups, observation_ids)):
        key = (trip_id, observation_id)
        if key not in selected or scores[row_index] > scores[selected[key]]:
            selected[key] = row_index
    return float(np.mean([y_true[index] == 1 for index in selected.values()]))


def _candidate_miss_rate(
    groups: np.ndarray,
    observation_ids: np.ndarray,
    candidate_ids: np.ndarray,
    reference_ids: np.ndarray,
) -> float:
    """Compute the share of observations whose reference candidate is absent."""
    candidates_by_observation: dict[tuple[Any, Any], set[Any]] = {}
    references_by_observation: dict[tuple[Any, Any], Any] = {}
    for trip_id, observation_id, candidate_id, reference_id in zip(
        groups, observation_ids, candidate_ids, reference_ids
    ):
        key = (trip_id, observation_id)
        if key in references_by_observation and references_by_observation[key] != reference_id:
            raise ValueError(
                "reference_ids must be consistent across candidate rows for each observation."
            )
        references_by_observation[key] = reference_id
        candidates_by_observation.setdefault(key, set()).add(candidate_id)

    misses = [
        reference_id not in candidates_by_observation[key]
        for key, reference_id in references_by_observation.items()
    ]
    return float(np.mean(misses))


def _update_model_comparison(
    comparison_path: Path,
    model_name: str,
    fold_frame: pd.DataFrame,
    estimator: BaseEstimator,
    n_splits: int,
    random_state: Optional[int],
) -> None:
    """Append or replace one model aggregate without deleting other models."""
    comparison_row: dict[str, Any] = {
        "model_name": model_name,
        "n_splits": n_splits,
        "random_state": random_state if random_state is not None else "",
    }
    try:
        comparison_row["model_config"] = json.dumps(
            estimator.get_params(deep=False),
            sort_keys=True,
            default=str,
        )
    except (AttributeError, TypeError, ValueError):
        comparison_row["model_config"] = ""

    for column in _METRIC_COLUMNS:
        values = pd.to_numeric(fold_frame[column], errors="coerce")
        comparison_row[f"{column}_mean"] = float(values.mean())
        comparison_row[f"{column}_std"] = float(values.std(ddof=1))

    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    if comparison_path.exists():
        existing = pd.read_csv(comparison_path)
        if "model_name" not in existing.columns:
            raise ValueError(
                f"{comparison_path} exists but does not contain a model_name column."
            )
        existing = existing.loc[existing["model_name"] != model_name].copy()
        combined = pd.concat(
            [existing, pd.DataFrame([comparison_row])],
            ignore_index=True,
        )
    else:
        combined = pd.DataFrame([comparison_row])
    combined.to_csv(comparison_path, index=False)


def evaluate_model(
    estimator: BaseEstimator,
    X: Any,
    y: Any,
    groups: Any,
    observation_ids: Any,
    model_name: str,
    *,
    n_splits: int = 5,
    random_state: Optional[int] = 42,
    results_dir: Any = Path("results") / "runs",
    candidate_ids: Any = None,
    reference_ids: Any = None,
) -> pd.DataFrame:
    """Trip-grouped CV of a binary candidate classifier; writes results_dir/<model_name>.csv."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty filename stem.")
    if (
        model_name in {".", ".."}
        or Path(model_name).name != model_name
        or any(character in model_name for character in '<>:"/\\|?*')
    ):
        raise ValueError("model_name must not contain path components.")
    if model_name.lower().endswith(".csv"):
        raise ValueError("Pass model_name without the .csv extension.")
    if candidate_ids is None and reference_ids is not None:
        raise ValueError("candidate_ids must be supplied when reference_ids are supplied.")
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2.")
    if random_state is not None and (
        not isinstance(random_state, int) or isinstance(random_state, bool)
    ):
        raise ValueError("random_state must be an integer or None.")

    y_array = _as_1d_array(y, "y")
    group_array = _as_1d_array(groups, "groups")
    observation_array = _as_1d_array(observation_ids, "observation_ids")
    row_count = len(y_array)
    if row_count == 0:
        raise ValueError("Evaluation inputs must contain at least one candidate row.")
    if getattr(X, "ndim", None) != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if X.shape[0] != row_count or X.shape[1] == 0:
        raise ValueError("X must have the same non-zero row count as y.")
    if len(group_array) != row_count or len(observation_array) != row_count:
        raise ValueError("X, y, groups, and observation_ids must have matching row counts.")

    _validate_non_missing(y_array, "y")
    _validate_non_missing(group_array, "groups")
    _validate_non_missing(observation_array, "observation_ids")
    _validate_hashable(group_array, "groups")
    _validate_hashable(observation_array, "observation_ids")
    if not _contains_only_binary_labels(y_array):
        raise ValueError("y must contain only binary labels 0 and 1.")
    distinct_trip_count = len(set(group_array.tolist()))
    if distinct_trip_count < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds the {distinct_trip_count} distinct trip IDs."
        )

    candidate_array = reference_array = None
    if candidate_ids is not None:
        candidate_array = _as_1d_array(candidate_ids, "candidate_ids")
        _validate_non_missing(candidate_array, "candidate_ids")
        _validate_hashable(candidate_array, "candidate_ids")
        if len(candidate_array) != row_count:
            raise ValueError("candidate_ids must align with X and y.")
    if reference_ids is not None:
        reference_array = _as_1d_array(reference_ids, "reference_ids")
        if len(reference_array) != row_count:
            raise ValueError("reference_ids must align with X and y.")
        _validate_non_missing(reference_array, "reference_ids")
        _validate_hashable(reference_array, "reference_ids")

    try:
        clone(estimator)
    except (TypeError, AttributeError) as exc:
        raise TypeError("estimator must be a cloneable scikit-learn-compatible estimator.") from exc
    if not callable(getattr(estimator, "fit", None)) or not callable(
        getattr(estimator, "predict", None)
    ):
        raise TypeError("estimator must provide callable fit and predict methods.")

    splitter = GroupKFold(n_splits=n_splits)
    try:
        splits = list(splitter.split(X, y_array, group_array))
    except (TypeError, ValueError) as exc:
        raise ValueError("groups must be valid, consistently typed trip identifiers.") from exc

    fold_rows: list[dict[str, Any]] = []
    for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_trips = set(group_array[train_indices].tolist())
        validation_trips = set(group_array[validation_indices].tolist())
        if train_trips.intersection(validation_trips):
            raise RuntimeError("GroupKFold produced overlapping trip IDs across a fold.")

        fold_estimator = clone(estimator)
        _seed_unset_random_states(fold_estimator, random_state)
        fit_start = time.perf_counter()
        fold_estimator.fit(_take_rows(X, train_indices), y_array[train_indices])
        training_time = time.perf_counter() - fit_start

        X_validation = _take_rows(X, validation_indices)
        y_validation = y_array[validation_indices]
        inference_start = time.perf_counter()
        predicted = np.asarray(fold_estimator.predict(X_validation))
        if predicted.ndim != 1 or len(predicted) != len(validation_indices):
            raise ValueError("predict must return one binary label per validation row.")
        if not _contains_only_binary_labels(predicted):
            raise ValueError("predict must return labels encoded as 0 and 1.")

        probabilities = None
        if callable(getattr(fold_estimator, "predict_proba", None)):
            try:
                raw_probabilities = fold_estimator.predict_proba(X_validation)
            except NotImplementedError:
                raw_probabilities = None
            if raw_probabilities is not None:
                probabilities = _positive_class_probability(
                    fold_estimator, raw_probabilities, len(validation_indices)
                )

        ranking_scores = probabilities
        inference_time = time.perf_counter() - inference_start

        tn, fp, fn, tp = confusion_matrix(
            y_validation, predicted, labels=[0, 1]
        ).ravel()
        fold_metrics: dict[str, Any] = {
            "record_type": "fold",
            "fold": fold_number,
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_validation, predicted)),
            "precision": float(
                precision_score(y_validation, predicted, pos_label=1, zero_division=0)
            ),
            "recall": float(
                recall_score(y_validation, predicted, pos_label=1, zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_validation, predicted, labels=[0, 1], average="macro", zero_division=0)
            ),
            "log_loss": (
                float(
                    log_loss(
                        y_validation,
                        np.column_stack((1.0 - probabilities, probabilities)),
                        labels=[0, 1],
                    )
                )
                if probabilities is not None
                else np.nan
            ),
            "top1_accuracy": (
                _observation_top1_accuracy(
                    y_validation,
                    ranking_scores,
                    group_array[validation_indices],
                    observation_array[validation_indices],
                )
                if ranking_scores is not None
                else np.nan
            ),
            "candidate_miss_rate": (
                _candidate_miss_rate(
                    group_array[validation_indices],
                    observation_array[validation_indices],
                    candidate_array[validation_indices],
                    reference_array[validation_indices],
                )
                if candidate_array is not None and reference_array is not None
                else np.nan
            ),
            "training_time_seconds": float(training_time),
            "inference_time_seconds": float(inference_time),
            "confusion_tn": int(tn),
            "confusion_fp": int(fp),
            "confusion_fn": int(fn),
            "confusion_tp": int(tp),
        }
        fold_rows.append(fold_metrics)

    result_rows = list(fold_rows)
    fold_frame = pd.DataFrame(fold_rows)
    for statistic in ("mean", "std"):
        aggregate: dict[str, Any] = {
            "record_type": statistic,
            "fold": np.nan,
            "model_name": model_name,
        }
        for column in _METRIC_COLUMNS:
            values = pd.to_numeric(fold_frame[column], errors="coerce")
            aggregate[column] = (
                float(values.mean())
                if statistic == "mean"
                else float(values.std(ddof=1))
            )
        result_rows.append(aggregate)

    results = pd.DataFrame(result_rows)
    output_path = Path(results_dir) / f"{model_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    comparison_path = Path(results_dir).parent / "model_comparison.csv"
    _update_model_comparison(
        comparison_path=comparison_path,
        model_name=model_name,
        fold_frame=fold_frame,
        estimator=estimator,
        n_splits=n_splits,
        random_state=random_state,
    )
    return results
