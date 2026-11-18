"""Optuna hyperparameter tuning for the existing PathSnap LightGBM model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from common.config import REPO_ROOT
from features.build_features import FEATURE_COLS
from models.common_eval import evaluate_model
from models.lgbm import make_model
from models.logreg import load_split, top1_accuracy


N_FOLDs = 5
SEED = 42
BROKEN_TRAIN_TRIPS = ["00000054", "00000033"]


def suggest_params(trial: optuna.Trial) -> dict:
    """Suggest exactly the requested Optuna search ranges."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int(
            "min_child_samples", 10, 200, log=True
        ),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
    }


def prepare_training_data() -> tuple[pd.DataFrame, list[str]]:
    """Reuse the existing loader and its train-only broken-trip filtering."""
    train = load_split("train")
    missing = [column for column in FEATURE_COLS if column not in train.columns]
    if missing:
        raise ValueError(f"Missing required LightGBM feature columns: {missing}")
    if "trip_id" not in train or "obs_idx" not in train or "arc_id" not in train:
        raise ValueError("Training data is missing trip_id, obs_idx, or arc_id")
    if "label" not in train:
        raise ValueError("Training data is missing the label column")

    labels = train["label"].to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must contain only 0 and 1")
    if np.unique(labels).size != 2:
        raise ValueError("Training data must contain both label classes")
    return train.reset_index(drop=True), list(FEATURE_COLS)


def trial_dataframe(study: optuna.study.Study) -> pd.DataFrame:
    """Convert Optuna trials to a stable, readable CSV representation."""
    rows: list[dict] = []
    param_names = (
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
    )
    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "objective_value": trial.value,
            "state": trial.state.name,
        }
        for name in param_names:
            row[name] = trial.params.get(name, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def save_history_plot(study: optuna.study.Study, output_path: Path) -> None:
    """Save an optimization history plot using Matplotlib's headless backend."""
    completed = [
        (trial.number, trial.value)
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    if completed:
        trial_numbers = [number for number, _ in completed]
        values = [value for _, value in completed]
        ax.plot(trial_numbers, values, marker="o", linewidth=1.5)
        running_best = np.maximum.accumulate(values)
        ax.plot(trial_numbers, running_best, linestyle="--", label="Running best")
        ax.legend()
        ax.set_xlabel("Trial")
        ax.set_ylabel("Mean validation Macro F1")
    else:
        ax.text(
            0.5,
            0.5,
            "No completed trials",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title("Optuna LightGBM optimization history")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_student_test(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> dict:
    """Fit one tuned model on train and evaluate once on student-test data."""
    train_start = time.perf_counter()
    model.fit(
        train[feature_columns].to_numpy(dtype=np.float32),
        train["label"].to_numpy(dtype=np.int8),
    )
    train_time = time.perf_counter() - train_start

    inference_start = time.perf_counter()
    probabilities = model.predict_proba(
        test[feature_columns].to_numpy(dtype=np.float32)
    )[:, 1]
    predictions = model.predict(test[feature_columns].to_numpy(dtype=np.float32))
    inference_time = time.perf_counter() - inference_start

    labels = test["label"].to_numpy(dtype=np.int8)
    top1 = top1_accuracy(test, probabilities)
    return {
        "top1_accuracy": float(top1),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "log_loss": float(
            log_loss(
                labels,
                np.clip(probabilities, 1e-15, 1.0 - 1e-15),
                labels=[0, 1],
            )
        ),
        "train_time_s": float(train_time),
        "inference_time_s": float(inference_time),
    }


def save_tuning_comparison(
    tuned_cv: pd.DataFrame,
    tuned_student: dict,
    output_path: Path,
) -> None:
    """Compare the tuned model with the existing untuned LightGBM outputs."""
    untuned_cv_path = REPO_ROOT / "results" / "runs" / "lgbm.csv"
    untuned_student_path = REPO_ROOT / "results" / "runs" / "lgbm_student_test.csv"

    if not untuned_cv_path.exists():
        raise FileNotFoundError(
            f"Untuned LightGBM CV results are unavailable: {untuned_cv_path}"
        )
    if not untuned_student_path.exists():
        raise FileNotFoundError(
            f"Untuned LightGBM student-test results are unavailable: "
            f"{untuned_student_path}"
        )

    untuned_cv = pd.read_csv(untuned_cv_path)
    untuned_mean = untuned_cv.loc[
        untuned_cv["record_type"] == "mean"
    ].iloc[0]
    untuned_student = pd.read_csv(untuned_student_path).iloc[0]

    tuned_mean = tuned_cv.set_index("record_type").loc["mean"]
    rows = [
        {
            "model": "lgbm",
            "cv_macro_f1_mean": float(untuned_mean["macro_f1"]),
            "student_test_top1": float(untuned_student["top1_accuracy"]),
            "student_test_macro_f1": float(untuned_student["macro_f1"]),
            "student_test_log_loss": float(untuned_student["log_loss"]),
            "student_test_inference_time_s": float(
                untuned_student["inference_time_s"]
            ),
        },
        {
            "model": "lgbm_tuned",
            "cv_macro_f1_mean": float(tuned_mean["macro_f1"]),
            "student_test_top1": tuned_student["top1_accuracy"],
            "student_test_macro_f1": tuned_student["macro_f1"],
            "student_test_log_loss": tuned_student["log_loss"],
            "student_test_inference_time_s": tuned_student["inference_time_s"],
        },
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main(trials: int = 40, timeout: int | None = None) -> optuna.study.Study:
    """Run Optuna tuning and the tuned LightGBM evaluation."""
    if trials < 1:
        raise ValueError("--trials must be at least 1")
    if timeout is not None and timeout < 1:
        raise ValueError("--timeout must be at least 1 second")

    train, feature_columns = prepare_training_data()
    X = train[feature_columns].to_numpy(dtype=np.float32)
    y = train["label"].to_numpy(dtype=np.int8)
    groups = train["trip_id"].to_numpy()

    splitter = GroupKFold(n_splits=N_FOLDs)
    splits = list(splitter.split(X, y, groups))
    for train_indices, validation_indices in splits:
        train_trips = set(groups[train_indices].tolist())
        validation_trips = set(groups[validation_indices].tolist())
        if train_trips.intersection(validation_trips):
            raise RuntimeError("GroupKFold produced overlapping trip IDs")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        fold_scores: list[float] = []
        for train_indices, validation_indices in splits:
            model = make_model(seed=SEED, **params)
            model.fit(X[train_indices], y[train_indices])
            predictions = model.predict(X[validation_indices])
            fold_scores.append(
                float(
                    f1_score(
                        y[validation_indices],
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                )
            )
        return float(np.mean(fold_scores))

    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="pathsnap_lgbm",
    )
    study.optimize(objective, n_trials=trials, timeout=timeout)

    trials_path = REPO_ROOT / "results" / "optuna_trials.csv"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    trial_dataframe(study).to_csv(trials_path, index=False)

    history_path = REPO_ROOT / "results" / "figures" / "optuna_history.png"
    save_history_plot(study, history_path)

    best_params = dict(study.best_trial.params)
    best_output = {
        "best_params": best_params,
        "best_cv_macro_f1": float(study.best_value),
        "n_trials": trials,
        "n_splits": N_FOLDs,
        "grouping_column": "trip_id",
        "random_seed": SEED,
        "excluded_train_trips": BROKEN_TRAIN_TRIPS,
    }
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = artifacts_dir / "best_params.json"
    best_params_path.write_text(
        json.dumps(best_output, indent=2),
        encoding="utf-8",
    )

    tuned_model = make_model(seed=SEED, **best_params)
    tuned_cv = evaluate_model(
        estimator=tuned_model,
        X=train[feature_columns],
        y=train["label"].astype(int),
        groups=train["trip_id"],
        observation_ids=train["obs_idx"],
        model_name="lgbm_tuned",
        n_splits=N_FOLDs,
        random_state=SEED,
        results_dir=REPO_ROOT / "results" / "runs",
        candidate_ids=train["arc_id"],
        reference_ids=(
            train["true_arc_id"]
            if "true_arc_id" in train.columns
            else None
        ),
    )

    test = load_split("student_test")
    tuned_student_model = make_model(seed=SEED, **best_params)
    tuned_student = evaluate_student_test(
        tuned_student_model,
        train,
        test,
        feature_columns,
    )

    joblib.dump(
        {
            "model": tuned_student_model,
            "feature_cols": feature_columns,
            "best_params": best_params,
            "best_cv_macro_f1": float(study.best_value),
        },
        artifacts_dir / "model.joblib",
    )

    save_tuning_comparison(
        tuned_cv,
        tuned_student,
        REPO_ROOT / "results" / "tuning_comparison.csv",
    )

    print(f"Trials CSV: {trials_path}")
    print(f"History plot: {history_path}")
    print(f"Best params: {best_params_path}")
    print(f"Best CV Macro F1: {study.best_value:.6f}")
    print(f"Tuned model: {artifacts_dir / 'model.joblib'}")
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune PathSnap LightGBM with Optuna and GroupKFold."
    )
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=None)
    arguments = parser.parse_args()
    main(trials=arguments.trials, timeout=arguments.timeout)
