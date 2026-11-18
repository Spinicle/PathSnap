"""Report figures that are not produced elsewhere. Every figure comes from repo data/results.
Partition is stated in every title. Output: results/figures/.
Run: python -m report.make_figures
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from sklearn.metrics import confusion_matrix, f1_score

from common.config import REPO_ROOT
from data_processing.loader import list_trips, load_trip
from features.candidates import project_trip
from models.logreg import BROKEN_TRAIN_TRIPS, load_split

FIGS = REPO_ROOT / "results" / "figures"
RUNS = REPO_ROOT / "results" / "runs"
ARTIFACTS = REPO_ROOT / "artifacts"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#008300", "#b0b0b0"
CONFUSION_MODELS = {"logreg": "logreg.joblib", "nb": "nb.joblib",
                    "random_forest": "random_forest.joblib", "lgbm": "lgbm.joblib",
                    "mlp": "mlp.joblib", "lgbm_tuned": "model.joblib"}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=150)
    plt.close(fig)
    print(f"wrote results/figures/{name}")


def sample_train_trips(n: int = 4) -> list[str]:
    trips = [t for t in list_trips("train") if t not in BROKEN_TRAIN_TRIPS]
    return list(np.random.default_rng(42).choice(trips, n, replace=False))


def gps_trajectories(trip_ids: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, tid in zip(axes.ravel(), trip_ids):
        trip = load_trip(tid, "train", with_route=False)
        P = project_trip(trip)
        x, y = P["px"] - P["px"][0], P["py"] - P["py"][0]
        ax.plot(x, y, color=BLUE, linewidth=1, alpha=0.6)
        ax.scatter(x, y, s=3, color=BLUE)
        ax.scatter([x[0]], [y[0]], s=40, color=GREEN, label="start", zorder=3)
        ax.scatter([x[-1]], [y[-1]], s=40, color=ORANGE, label="end", zorder=3)
        ax.set_title(f"trip {tid} ({len(x)} GPS points)", fontsize=10)
        ax.set_xlabel("Easting from start (m)")
        ax.set_ylabel("Northing from start (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle("GPS trajectories, 4 train trips (seed 42), per-trip UTM projection")
    save(fig, "eda_gps_trajectories.png")


def smallest_extent(trip_ids: list[str]) -> str:
    """Sampled trip with the smallest GPS bounding box (most readable network plot)."""
    def extent(tid):
        P = project_trip(load_trip(tid, "train", with_route=False))
        return np.ptp(P["px"]) * np.ptp(P["py"])
    return min(trip_ids, key=extent)


def road_network(tid: str, margin_m: float = 300.0) -> None:
    trip = load_trip(tid, "train", with_route=True)
    P = project_trip(trip)
    ox, oy = P["px"][0], P["py"][0]  # plot in metres from the first GPS point
    px, py, nx, ny = P["px"] - ox, P["py"] - oy, P["nx"] - ox, P["ny"] - oy
    x0, x1 = px.min() - margin_m, px.max() + margin_m
    y0, y1 = py.min() - margin_m, py.max() + margin_m
    u = trip["arcs"]["u"].to_numpy()
    v = trip["arcs"]["v"].to_numpy()
    inside = lambda i: (nx[i] >= x0) & (nx[i] <= x1) & (ny[i] >= y0) & (ny[i] <= y1)
    keep = inside(u) | inside(v)
    segs = np.stack([np.c_[nx[u[keep]], ny[u[keep]]], np.c_[nx[v[keep]], ny[v[keep]]]], axis=1)
    route = np.asarray(trip["route"], dtype=int)
    rsegs = np.stack([np.c_[nx[u[route]], ny[u[route]]], np.c_[nx[v[route]], ny[v[route]]]], axis=1)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_collection(LineCollection(segs, colors=GREY, linewidths=0.6, label="road network"))
    ax.add_collection(LineCollection(rsegs, colors=GREEN, linewidths=4, label="reference route"))
    ax.scatter(px, py, s=3, color=BLUE, alpha=0.6, label="GPS observations", zorder=3)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting from first GPS point (m, UTM)")
    ax.set_ylabel("Northing from first GPS point (m, UTM)")
    ax.legend(frameon=True, loc="best")
    ax.set_title(f"Road network around train trip {tid}\n{keep.sum():,} of {len(u):,} arcs shown "
                 f"(GPS bbox + {margin_m:.0f} m), {len(route)} route arcs, {len(px)} GPS points",
                 fontsize=10)
    save(fig, "eda_road_network.png")


def missing_values() -> None:
    inv = pd.read_csv(REPO_ROOT / "results" / "eda_inventory.csv")
    cols = ["missing_lon", "missing_lat", "missing_t", "invalid_gps", "isolated_nodes"]
    g = inv.groupby("partition").agg(trips=("trip_id", "count"), observations=("observations", "sum"),
                                     **{c: (c, "sum") for c in cols})
    g = g.loc[[p for p in ("train", "student_test") if p in g.index]]
    print("Missing / invalid counts per partition (results/eda_inventory.csv):")
    print(g.to_string())
    fig, ax = plt.subplots(figsize=(10, 1.8))
    ax.axis("off")
    t = ax.table(cellText=g.reset_index().astype(str).values,
                 colLabels=["partition"] + list(g.columns), loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)
    ax.set_title("Missing and invalid values per column, train and student_test partitions "
                 "(summed over trips)", fontsize=10)
    save(fig, "eda_missing_values.png")


def reported_test_f1(name: str) -> float:
    path = RUNS / f"{name}_student_test.csv"
    if path.exists():
        return float(pd.read_csv(path)["macro_f1"].iloc[0])
    t = pd.read_csv(REPO_ROOT / "results" / "tuning_comparison.csv").set_index("model")
    return float(t.loc[name, "student_test_macro_f1"])


def confusion_matrices() -> None:
    test = load_split("student_test")
    y = test["label"].astype(int).to_numpy()
    for name, fname in CONFUSION_MODELS.items():
        try:
            art = joblib.load(ARTIFACTS / fname)
        except Exception as exc:  # e.g. RF joblib missing or too large to load
            print(f"skip confusion_{name}.png: could not load artifacts/{fname} ({exc})")
            continue
        p = art["model"].predict_proba(test[art["feature_cols"]])[:, 1]
        pred = (p >= 0.5).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        f1 = f1_score(y, pred, average="macro")
        ref = reported_test_f1(name)
        print(f"{name}: macro F1 from matrix {f1:.4f} | reported {ref:.4f} | diff {abs(f1 - ref):.5f}")
        if abs(f1 - ref) > 0.001:
            raise SystemExit(f"STOP: {name} macro F1 differs from reported value by > 0.001")
        pct = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({pct[i, j]:.1%})", ha="center", va="center",
                        color="white" if pct[i, j] > 0.5 else "black", fontsize=11)
        ax.set_xticks([0, 1], ["0: not true arc", "1: true arc"])
        ax.set_yticks([0, 1], ["0: not true arc", "1: true arc"])
        ax.set_xlabel("Predicted label (P >= 0.5)")
        ax.set_ylabel("True label")
        ax.set_title(f"{name}, student_test candidate rows\nmacro F1 {f1:.4f}", fontsize=10)
        save(fig, f"confusion_{name}.png")


def tuning_comparison() -> None:
    t = pd.read_csv(REPO_ROOT / "results" / "tuning_comparison.csv").set_index("model")
    metrics = [("cv_macro_f1_mean", "CV macro F1\n(train, GroupKFold 5)"),
               ("student_test_top1", "Top-1\n(student_test)"),
               ("student_test_macro_f1", "Macro F1\n(student_test)"),
               ("student_test_log_loss", "Log loss\n(student_test)")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [3, 1]})
    x, w = np.arange(len(metrics)), 0.38
    for k, (model, color, label) in enumerate([("lgbm", BLUE, "untuned"),
                                                ("lgbm_tuned", ORANGE, "tuned (Optuna)")]):
        vals = [t.loc[model, m] for m, _ in metrics]
        b = ax.bar(x + (k - 0.5) * w, vals, w, color=color, label=label, edgecolor="white")
        ax.bar_label(b, fmt="%.4f", fontsize=8, padding=2)
    ax.set_xticks(x, [lbl for _, lbl in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.set_title("LightGBM untuned vs tuned")
    inf = [t.loc["lgbm", "student_test_inference_time_s"], t.loc["lgbm_tuned", "student_test_inference_time_s"]]
    b = ax2.bar([0, 1], inf, 0.6, color=[BLUE, ORANGE], edgecolor="white")
    ax2.bar_label(b, fmt="%.2f s", fontsize=8, padding=2)
    ax2.set_xticks([0, 1], ["untuned", "tuned"])
    ax2.set_ylabel("Seconds")
    ax2.set_title("Inference time\n(student_test, all rows)")
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.3)
        a.set_axisbelow(True)
    save(fig, "tuning_comparison.png")


def hmm_comparison() -> None:
    """Per-trip classifier-only vs HMM, read from results/hmm_comparison.csv (hmm/evaluate.py)."""
    t = pd.read_csv(REPO_ROOT / "results" / "hmm_comparison.csv", dtype={"trip_id": str})
    trips = t[t["record_type"] == "trip"]
    mean = t[t["record_type"] == "mean"].iloc[0]
    labels = trips["trip_id"].tolist() + ["mean"]
    x, w = np.arange(len(labels)), 0.38
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, (cls_col, hmm_col, name) in zip(axes, [
            ("classifier_only_top1", "hmm_top1", "Top-1 accuracy"),
            ("classifier_only_route_continuity", "hmm_route_continuity", "Route continuity")]):
        for k, (col, color, lbl) in enumerate([(cls_col, BLUE, "classifier only (argmax)"),
                                               (hmm_col, ORANGE, "classifier + HMM/Viterbi")]):
            vals = trips[col].tolist() + [mean[col]]
            b = ax.bar(x + (k - 0.5) * w, vals, w, color=color, label=lbl, edgecolor="white")
            ax.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
        ax.set_ylabel(name)
        ax.set_ylim(0, 1.15)
        ax.axvline(len(labels) - 1.5, color="black", linewidth=0.8, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper center", ncol=2)
    axes[0].set_title("Classifier-only vs HMM per trip, student_test (tuned LightGBM; "
                      "mean = unweighted over trips)")
    axes[1].set_xticks(x, labels, rotation=30)
    axes[1].set_xlabel("student_test trip")
    save(fig, "hmm_comparison.png")


def hmm_beta_sweep() -> None:
    """Top-1 vs beta from results/hmm_beta_sweep.csv; chosen beta read from config.yaml."""
    from common.config import CFG

    chosen = CFG["hmm"]["beta"]
    t = pd.read_csv(REPO_ROOT / "results" / "hmm_beta_sweep.csv", dtype={"trip_id": str})
    trips = t[t["record_type"] == "trip"]
    mean = t[t["record_type"] == "mean"].sort_values("beta")
    fig, ax = plt.subplots(figsize=(8, 5))
    for tid, g in trips.groupby("trip_id"):
        g = g.sort_values("beta")
        ax.plot(g["beta"], g["top1_accuracy"], color=GREY, linewidth=1, marker="o", markersize=3)
        ax.annotate(tid, (g["beta"].iloc[-1], g["top1_accuracy"].iloc[-1]), fontsize=7,
                    xytext=(4, 0), textcoords="offset points", va="center", color="#555555")
    ax.plot(mean["beta"], mean["top1_accuracy"], color=BLUE, linewidth=2.5, marker="o",
            label="mean over trips")
    ax.axvline(chosen, color=ORANGE, linestyle="--", linewidth=1.5, label=f"chosen beta = {chosen}")
    ax.set_xscale("log")
    ax.set_xticks(mean["beta"], [str(int(b)) for b in mean["beta"]])
    ax.set_xlabel("beta (transition scale, m), log scale")
    ax.set_ylabel("HMM top-1 accuracy")
    ax.set_title(f"HMM beta sweep, train trips {', '.join(sorted(trips['trip_id'].unique()))}\n"
                 "(grey = per trip, tuned LightGBM probabilities)", fontsize=10)
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    save(fig, "hmm_beta_sweep.png")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["eda", "models", "hmm"], default=None,
                    help="eda: trajectories/network/missing; models: tuning + confusion; hmm: HMM plots")
    only = ap.parse_args().only
    FIGS.mkdir(parents=True, exist_ok=True)
    if only in (None, "eda"):
        trips = sample_train_trips()
        print("sampled train trips (seed 42):", trips)
        gps_trajectories(trips)
        road_network(smallest_extent(trips))
        missing_values()
    if only in (None, "models"):
        tuning_comparison()
        confusion_matrices()
    if only in (None, "hmm"):
        hmm_comparison()
        hmm_beta_sweep()
