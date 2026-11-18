from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_NAMES = {
    "logreg": "Logistic Regression",
    "nb": "Gaussian Naive Bayes",
    "random_forest": "Random Forest",
    "lgbm": "LightGBM",
    "mlp": "MLP",
}

METRICS = (
    ("accuracy", "accuracy"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("macro_f1", "macro_f1"),
    ("log_loss", "log_loss"),
    ("top1_accuracy", "top1_accuracy"),
    ("candidate_miss_rate", "candidate_miss_rate"),
    ("training_time_seconds", "training_time_seconds"),
    ("inference_time_seconds", "inference_time_seconds"),
)


def load_fold_results(runs_dir: Path) -> dict[str, pd.DataFrame]:
    available: dict[str, pd.DataFrame] = {}
    for model_key, display_name in MODEL_NAMES.items():
        path = runs_dir / f"{model_key}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "record_type" in frame.columns:
            frame = frame.loc[frame["record_type"] == "fold"].copy()
        if frame.empty:
            continue
        required = {"fold", *[column for column, _ in METRICS]}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        frame["model_name"] = display_name
        frame["model_key"] = model_key
        available[model_key] = frame
    return available


def aggregate_results(
    fold_results: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    for model_key, display_name in MODEL_NAMES.items():
        frame = fold_results.get(model_key)
        if frame is None:
            continue
        row: dict[str, object] = {
            "model_name": display_name,
            "n_splits": int(frame["fold"].nunique()),
        }
        for column, _ in METRICS:
            values = pd.to_numeric(frame[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
            )
        rows.append(row)
    if not rows:
        raise ValueError("No model fold CSV files are available under results/runs.")
    return pd.DataFrame(rows), fold_results


def update_comparison_csv(
    comparison_path: Path,
    aggregates: pd.DataFrame,
    model_names: Iterable[str],
) -> None:
    """Replace evaluated models while preserving unrelated existing model rows."""
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    if comparison_path.exists():
        existing = pd.read_csv(comparison_path)
        if "model_name" not in existing.columns:
            raise ValueError(
                f"{comparison_path} does not contain a model_name column."
            )
        display_to_key = {display: key for key, display in MODEL_NAMES.items()}
        names_to_remove = set(model_names)
        names_to_remove.update(
            display_to_key[name]
            for name in model_names
            if name in display_to_key
        )
        existing = existing.loc[
            ~existing["model_name"].astype(str).isin(names_to_remove)
        ].copy()
        combined = pd.concat([existing, aggregates], ignore_index=True)
    else:
        combined = aggregates

    combined = combined.loc[
        ~combined["model_name"].duplicated(keep="last")
    ].reset_index(drop=True)
    combined.to_csv(comparison_path, index=False)


def plot_metric(
    aggregates: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    percentage: bool = False,
) -> None:
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    if mean_column not in aggregates.columns:
        return

    frame = aggregates.loc[
        pd.to_numeric(aggregates[mean_column], errors="coerce").notna()
    ].copy()
    if frame.empty:
        return

    names = frame["model_name"].astype(str).tolist()
    means = pd.to_numeric(frame[mean_column], errors="coerce").to_numpy(dtype=float)
    stds = pd.to_numeric(frame[std_column], errors="coerce").to_numpy(dtype=float)
    if percentage:
        means = means * 100.0
        stds = stds * 100.0
        ylabel = f"{ylabel} (%)"

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color="#4C78A8",
        alpha=0.9,
        ecolor="#333333",
    )
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in means], padding=3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_plots(aggregates: pd.DataFrame, plots_dir: Path) -> list[Path]:
    specs = (
        ("macro_f1", "Macro F1 comparison", "Macro F1", "macro_f1.png", True),
        ("top1_accuracy", "Top-1 accuracy comparison", "Top-1 accuracy", "top1_accuracy.png", True),
        ("accuracy", "Accuracy comparison", "Accuracy", "accuracy.png", True),
        ("log_loss", "Log loss comparison", "Log loss", "log_loss.png", False),
        (
            "training_time_seconds",
            "Training time comparison",
            "Training time (seconds)",
            "training_time.png",
            False,
        ),
        (
            "inference_time_seconds",
            "Inference time comparison",
            "Inference time (seconds)",
            "inference_time.png",
            False,
        ),
    )
    generated: list[Path] = []
    for metric, title, ylabel, filename, percentage in specs:
        output = plots_dir / filename
        plot_metric(aggregates, metric, title, ylabel, output, percentage)
        if output.exists():
            generated.append(output)
    return generated


def print_ranking(aggregates: pd.DataFrame) -> None:
    def available(metric: str) -> pd.DataFrame:
        column = f"{metric}_mean"
        return aggregates.loc[
            pd.to_numeric(aggregates[column], errors="coerce").notna()
        ]

    for metric, title, ascending in (
        ("macro_f1", "Highest mean Macro F1", False),
        ("top1_accuracy", "Highest mean Top-1 accuracy", False),
        ("log_loss", "Lowest mean log loss", True),
        ("inference_time_seconds", "Fastest mean inference", True),
    ):
        frame = available(metric)
        if frame.empty:
            print(f"{title}: unavailable")
            continue
        frame = frame.assign(
            _value=pd.to_numeric(frame[f"{metric}_mean"], errors="coerce")
        ).sort_values("_value", ascending=ascending)
        row = frame.iloc[0]
        print(f"{title}: {row['model_name']} ({row['_value']:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare existing Phase 1 model fold-level CSV outputs."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("results") / "runs",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("results") / "model_comparison.csv",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("results") / "plots",
    )
    args = parser.parse_args()

    fold_results = load_fold_results(args.runs_dir)
    aggregates, _ = aggregate_results(fold_results)
    display_names = set(aggregates["model_name"].astype(str))
    update_comparison_csv(args.comparison, aggregates, display_names)
    generated = generate_plots(aggregates, args.plots_dir)
    print(aggregates.to_string(index=False))
    print_ranking(aggregates)
    print(f"Comparison CSV: {args.comparison}")
    print("Plots:")
    for plot in generated:
        print(plot)


if __name__ == "__main__":
    main()
