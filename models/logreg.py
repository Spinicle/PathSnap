"""Logistic regression baseline (Owner: A).
GroupKFold CV via models/common_eval.py, then fit on all of train and score student_test.
Run: python -m models.logreg
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.config import REPO_ROOT
from models.common_eval import evaluate_model

CACHE = REPO_ROOT / "cache"
RUNS = REPO_ROOT / "results" / "runs"
ARTIFACTS = REPO_ROOT / "artifacts"
BROKEN_TRAIN_TRIPS = ["00000054", "00000033"]  # GPS ~120-140 m from route: labels unreliable


def load_split(split: str) -> pd.DataFrame:
    df = pd.read_parquet(CACHE / f"features_{split}.parquet")
    lab = pd.read_parquet(CACHE / f"labels_{split}.parquet")[["trip_id", "obs_idx", "true_arc_id"]]
    df = df.merge(lab, on=["trip_id", "obs_idx"], how="left")
    if split == "train":
        n0 = len(df)
        df = df[~df["trip_id"].isin(BROKEN_TRAIN_TRIPS)].reset_index(drop=True)
        print(f"train: dropped trips {BROKEN_TRAIN_TRIPS} -> removed {n0 - len(df)} of {n0} rows")
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]


def top1_accuracy(df: pd.DataFrame, p: np.ndarray) -> float:
    s = df[["trip_id", "obs_idx", "label"]].assign(p=p)
    best = s.loc[s.groupby(["trip_id", "obs_idx"])["p"].idxmax()]
    return float(best["label"].mean())


def run_baseline(name: str, estimator) -> Pipeline:
    train, test = load_split("train"), load_split("student_test")
    cols = feature_cols(train)
    print(f"{name}: {len(cols)} features, train rows {len(train)}, student_test rows {len(test)}")

    cv = evaluate_model(estimator=estimator, X=train[cols], y=train["label"].astype(int),
                        groups=train["trip_id"], observation_ids=train["obs_idx"],
                        model_name=name, results_dir=RUNS,
                        candidate_ids=train["arc_id"], reference_ids=train["true_arc_id"])
    m, s = cv.set_index("record_type").loc["mean"], cv.set_index("record_type").loc["std"]
    print(f"CV macro F1   {m.macro_f1:.4f} ± {s.macro_f1:.4f}")
    print(f"CV top-1 acc  {m.top1_accuracy:.4f} ± {s.top1_accuracy:.4f}")

    t0 = time.perf_counter()
    estimator.fit(train[cols], train["label"].astype(int))
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = estimator.predict_proba(test[cols])[:, 1]
    pred = estimator.predict(test[cols])
    infer_s = time.perf_counter() - t0
    y = test["label"].astype(int).to_numpy()
    res = dict(model=name, top1_accuracy=top1_accuracy(test, p),
               macro_f1=f1_score(y, pred, average="macro"),
               log_loss=log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
               train_time_s=train_s, inference_time_s=infer_s,
               cv_macro_f1_mean=m.macro_f1, cv_macro_f1_std=s.macro_f1)
    print(f"student_test  top-1 {res['top1_accuracy']:.4f} | macro F1 {res['macro_f1']:.4f} | "
          f"log loss {res['log_loss']:.4f} | train {train_s:.2f}s | inference {infer_s:.3f}s")
    RUNS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res]).to_csv(RUNS / f"{name}_student_test.csv", index=False)
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump({"model": estimator, "feature_cols": cols}, ARTIFACTS / f"{name}.joblib")
    return estimator


if __name__ == "__main__":
    model = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=1000,
                                                 random_state=42))])
    model = run_baseline("logreg", model)
    cols = joblib.load(ARTIFACTS / "logreg.joblib")["feature_cols"]
    coef = model.named_steps["clf"].coef_[0]
    out = pd.DataFrame({"feature": cols, "coef": coef, "abs_coef": np.abs(coef)})
    out = out.sort_values("abs_coef", ascending=False)
    out.to_csv(RUNS / "logreg_coefs.csv", index=False)
    print(out.to_string(index=False))
