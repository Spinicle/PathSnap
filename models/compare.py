"""Final comparison table (report section 3.1) + student-test plots.
Complements models/compare_models.py (Vyshali), which covers the CV metrics and CV plots
(results/model_comparison.csv, results/plots/). This script does not touch those files.
Reads CV macro F1 from results/runs/<model>.csv and student-test numbers from
results/runs/<model>_student_test.csv; for lgbm_tuned the student-test row comes from
results/tuning_comparison.csv (written by optimisation/optuna_search.py).
A model is included only if its CSVs exist, so re-running picks up new results.
Route continuity = classifier-only argmax per GPS point on student_test, scored with
hmm.metrics.route_continuity; NaN (with a warning) until that module exists.
Run: python -m models.compare
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import REPO_ROOT
from models.logreg import CACHE

RUNS = REPO_ROOT / "results" / "runs"
FIGS = REPO_ROOT / "results" / "figures"
ARTIFACTS = REPO_ROOT / "artifacts"
OUT = REPO_ROOT / "results" / "model_comparison_final.csv"

MODELS = ["logreg", "nb", "random_forest", "lgbm", "mlp", "lgbm_tuned"]
FAMILY = {"logreg": "Linear", "nb": "Probabilistic (generative)",
          "random_forest": "Bagging ensemble (trees)", "lgbm": "Gradient boosting (trees)",
          "mlp": "Neural network", "lgbm_tuned": "Gradient boosting (trees), Optuna-tuned"}
ARTIFACT = {m: f"{m}.joblib" for m in MODELS} | {"lgbm_tuned": "model.joblib"}
KEY_PARAMS = {"logreg": ["C", "class_weight"], "nb": ["var_smoothing"],
              "random_forest": ["n_estimators", "max_depth", "min_samples_leaf", "max_features",
                                "class_weight"],
              "lgbm": ["n_estimators", "learning_rate", "num_leaves", "min_child_samples"],
              "mlp": ["hidden_layer_sizes", "alpha", "learning_rate_init", "max_iter"],
              "lgbm_tuned": ["n_estimators", "learning_rate", "num_leaves", "min_child_samples"]}
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load_artifact(m: str):
    path = ARTIFACTS / ARTIFACT[m]
    return joblib.load(path) if path.exists() else None


def key_config(m: str, art) -> str:
    if art is None:
        return ""
    est = art["model"]
    est = est.steps[-1][1] if hasattr(est, "steps") else est
    params = est.get_params()
    return ", ".join(f"{k}={params[k]}" for k in KEY_PARAMS[m] if k in params)


def route_continuity_score(art, test: pd.DataFrame, route_continuity) -> float:
    from data_processing.loader import load_trip

    p = art["model"].predict_proba(test[art["feature_cols"]])[:, 1]
    s = test[["trip_id", "obs_idx", "arc_id"]].assign(p=p)
    best = s.loc[s.groupby(["trip_id", "obs_idx"])["p"].idxmax()]
    scores = []
    for tid, g in best.groupby("trip_id"):
        trip = load_trip(tid, "student_test", with_route=False)
        arcs: list = [None] * len(trip["track"])
        for i, a in zip(g["obs_idx"], g["arc_id"]):
            arcs[int(i)] = int(a)
        scores.append(route_continuity(arcs, trip))
    return float(np.mean(scores))


def student_test_row(m: str) -> dict | None:
    """top1_accuracy, macro_f1, log_loss, inference_time_s on student_test, or None."""
    path = RUNS / f"{m}_student_test.csv"
    if path.exists():
        return pd.read_csv(path).iloc[0].to_dict()
    tuning = REPO_ROOT / "results" / "tuning_comparison.csv"
    if m == "lgbm_tuned" and tuning.exists():
        t = pd.read_csv(tuning).set_index("model")
        if m in t.index:
            r = t.loc[m]
            return {"top1_accuracy": r["student_test_top1"], "macro_f1": r["student_test_macro_f1"],
                    "log_loss": r["student_test_log_loss"],
                    "inference_time_s": r["student_test_inference_time_s"]}
    return None


def build_table() -> pd.DataFrame:
    try:
        from hmm.metrics import route_continuity
    except ImportError:
        route_continuity = None
        print("WARNING: hmm.metrics.route_continuity not found (git pull?). "
              "route_continuity column = NaN; re-run compare.py once it is pushed.")
    test = pd.read_parquet(CACHE / "features_student_test.parquet") if route_continuity else None

    rows = []
    for m in MODELS:
        cv_path = RUNS / f"{m}.csv"
        st = student_test_row(m)
        if not cv_path.exists() or st is None:
            print(f"skip {m}: missing CV or student-test results")
            continue
        cv = pd.read_csv(cv_path).set_index("record_type")
        art = load_artifact(m)
        rc = np.nan
        if route_continuity is not None:
            if art is None:
                print(f"WARNING: {m}: artifacts/{ARTIFACT[m]} missing, route_continuity = NaN")
            else:
                rc = route_continuity_score(art, test, route_continuity)
        mean, std = cv.loc["mean", "macro_f1"], cv.loc["std", "macro_f1"]
        rows.append({"model": m, "algorithm_family": FAMILY[m], "key_config": key_config(m, art),
                     "cv_macro_f1": f"{mean:.4f} ± {std:.4f}",
                     "cv_macro_f1_mean": mean, "cv_macro_f1_std": std,
                     "student_test_top1": st["top1_accuracy"],
                     "route_continuity_classifier_only": rc,
                     "student_test_macro_f1": st["macro_f1"],
                     "student_test_log_loss": st["log_loss"],
                     "student_test_inference_time_s": st["inference_time_s"]})
    return pd.DataFrame(rows)


def _bars(ax, x, vals, width, color, label=None, err=None):
    bars = ax.bar(x, vals, width, color=color, label=label, yerr=err,
                  capsize=4 if err is not None else 0, edgecolor="white", linewidth=1)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)


def _finish(ax, fig, models, title, ylabel, path, legend=None):
    ax.set_xticks(range(len(models)), models)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(frameon=False, loc=legend, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")


def plot(df: pd.DataFrame) -> None:
    """Student-test plots only; CV plots are made by models/compare_models.py."""
    FIGS.mkdir(parents=True, exist_ok=True)
    models, x, w = df["model"].tolist(), np.arange(len(df)), 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bars(ax, x - w / 2, df["student_test_top1"], w, BLUE, "Top-1 accuracy")
    _bars(ax, x + w / 2, df["student_test_macro_f1"], w, ORANGE, "Macro F1")
    ax.set_ylim(0, 1.15)
    _finish(ax, fig, models, "Student-test top-1 accuracy and macro F1", "Score",
            FIGS / "student_test_top1_f1.png", legend="upper center")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bars(ax, x, df["student_test_log_loss"], 0.6, BLUE)
    _finish(ax, fig, models, "Student-test log loss (lower is better)", "Log loss",
            FIGS / "student_test_log_loss.png")


if __name__ == "__main__":
    table = build_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.drop(columns=["key_config", "algorithm_family"]).to_string(index=False))
    plot(table)
