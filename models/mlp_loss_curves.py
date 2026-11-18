from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BROKEN_TRAIN_TRIPS = ["00000054", "00000033"]
EPOCHS = 100


def main() -> None:
    df = pd.read_parquet(ROOT / "cache" / "features_train.parquet")
    df = df[~df["trip_id"].isin(BROKEN_TRAIN_TRIPS)].reset_index(drop=True)
    cols = [c for c in df.columns if c.startswith("f_")]
    X, y, g = df[cols].to_numpy(np.float32), df["label"].astype(int).to_numpy(), df["trip_id"].to_numpy()
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, g))
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1e-4,
                        learning_rate_init=1e-3, batch_size=1024, random_state=42)
    rows = []
    for epoch in range(1, EPOCHS + 1):
        clf.partial_fit(Xtr, y[tr], classes=np.array([0, 1]))
        rows.append({"epoch": epoch,
                     "train_log_loss": log_loss(y[tr], clf.predict_proba(Xtr)[:, 1], labels=[0, 1]),
                     "val_log_loss": log_loss(y[va], clf.predict_proba(Xva)[:, 1], labels=[0, 1])})
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    (ROOT / "results" / "runs").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results" / "runs" / "mlp_loss_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(out["epoch"], out["train_log_loss"], label=f"training ({len(set(g[tr]))} trips)")
    ax.plot(out["epoch"], out["val_log_loss"], label=f"validation ({len(set(g[va]))} held-out trips)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Log loss")
    ax.set_title("MLP training and validation loss (train partition, split by trip)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "mlp_train_val_loss.png", dpi=150)
    print("best val epoch", int(out["val_log_loss"].idxmin()) + 1, "min val", out["val_log_loss"].min())


if __name__ == "__main__":
    main()
