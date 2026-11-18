"""Random Forest: CV + student_test through run_baseline, same pipeline as the other models.
Same hyperparameters as models/rf.py (the original CV-only exploration run), except
n_jobs=-1 instead of 2. With random_state=42 fixed, n_jobs only changes speed, not results.
Run: python -m models.rf_test
"""
from sklearn.ensemble import RandomForestClassifier

from models.logreg import run_baseline


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


if __name__ == "__main__":
    run_baseline("random_forest", make_model())
