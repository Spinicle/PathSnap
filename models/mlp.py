"""MLP model using the real PathSnap feature tables.

Run: python -m models.mlp
"""
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.logreg import run_baseline


def make_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=1024,
            max_iter=100,
            early_stopping=False,
            random_state=42,
        )),
    ])


if __name__ == "__main__":
    model = run_baseline("mlp", make_model())

    # Save the loss curve from the final fit on the training data.
    loss_curve = model.named_steps["clf"].loss_curve_
    output = Path("results") / "figures" / "mlp_loss_curve.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_curve) + 1), loss_curve)
    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.title("MLP training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    print(f"Saved MLP loss curve to {output}")