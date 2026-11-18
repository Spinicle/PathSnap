from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common.config import REPO_ROOT
from data_processing.loader import list_trips, load_trip
from features.build_features import FEATURE_COLS
from hmm.metrics import route_continuity
from hmm.viterbi import decode

CACHE = REPO_ROOT / "cache"
ARTIFACTS = REPO_ROOT / "artifacts"
RESULTS = REPO_ROOT / "results"
BETA_VALUES = (10, 25, 50, 100, 200)
BAD_TRAIN_TRIPS = {"00000054", "00000033"}


def load_model():
    """Load tuned model when present; otherwise use baseline LightGBM."""
    path = ARTIFACTS / "model.joblib"
    if not path.exists():
        path = ARTIFACTS / "lgbm.joblib"
    if not path.exists():
        raise FileNotFoundError("Need artifacts/model.joblib or artifacts/lgbm.joblib")

    saved = joblib.load(path)
    if isinstance(saved, dict) and "model" in saved:
        model = saved["model"]
        columns = saved.get("feature_cols") or FEATURE_COLS
    else:
        model = saved
        columns = FEATURE_COLS

    print(f"Using model: {path.name}")
    return model, list(columns)


def get_probabilities(model, columns, candidates):
    if candidates.empty:
        return np.asarray([], dtype=float)

    missing = [column for column in columns if column not in candidates.columns]
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")

    raw = np.asarray(
        model.predict_proba(candidates[columns].to_numpy(dtype=np.float32)),
        dtype=float,
    )
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    positive = np.flatnonzero(classes == 1)
    if raw.ndim != 2 or len(positive) != 1:
        raise ValueError("Model does not provide probabilities for class 1")
    return raw[:, positive[0]]


def classifier_arcs(candidates, probabilities, point_count):
    """Highest-probability road candidate per GPS point; missing points stay None."""
    predictions = [None] * point_count
    if candidates.empty:
        return predictions

    scored = candidates[["obs_idx", "arc_id"]].copy()
    scored["_prob"] = probabilities
    for obs_idx, group in scored.groupby("obs_idx", sort=False):
        best = group["_prob"].idxmax()
        predictions[int(obs_idx)] = int(scored.at[best, "arc_id"])
    return predictions


def top1(predictions, reference_arcs):
    """Count None predictions as incorrect; average across all GPS points."""
    if len(reference_arcs) == 0:
        return float("nan")
    correct = [
        prediction is not None and int(prediction) == int(reference)
        for prediction, reference in zip(predictions, reference_arcs)
    ]
    return float(np.mean(correct))


def load_reference_arcs(split):
    labels = pd.read_parquet(CACHE / f"labels_{split}.parquet")
    return {
        trip_id: group.sort_values("obs_idx")["true_arc_id"].astype(int).tolist()
        for trip_id, group in labels.groupby("trip_id")
    }


def load_candidate_table(split):
    path = CACHE / f"features_{split}.parquet"
    return pd.read_parquet(path)


def predict_trip(model, columns, candidates, trip_id, split, beta=None):
    trip_candidates = (
        candidates.loc[candidates["trip_id"] == trip_id]
        .copy()
        .reset_index(drop=True)
    )
    trip = load_trip(trip_id, split, with_route=False)
    probabilities = get_probabilities(model, columns, trip_candidates)
    classifier_only = classifier_arcs(
        trip_candidates, probabilities, len(trip["track"])
    )

    if "proj_frac" not in trip_candidates and "f_proj_frac" in trip_candidates:
        trip_candidates["proj_frac"] = trip_candidates["f_proj_frac"]

    hmm_prediction = decode(
        trip_candidates,
        probabilities,
        trip,
        beta_m=beta,
    )
    return trip, classifier_only, hmm_prediction


def save_beta_to_config(beta):
    """Change only the beta value under the hmm section, preserving other text."""
    path = REPO_ROOT / "config.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    in_hmm = False

    for index, line in enumerate(lines):
        if line.strip() == "hmm:":
            in_hmm = True
            continue
        if in_hmm and line and not line[0].isspace():
            break
        if in_hmm and line.lstrip().startswith("beta:"):
            indent = line[: len(line) - len(line.lstrip())]
            comment = ""
            if "#" in line:
                comment = "  #" + line.split("#", 1)[1]
            lines[index] = f"{indent}beta: {beta}{comment}"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"Set hmm.beta = {beta} in config.yaml")
            return

    raise ValueError("Could not find beta under hmm: in config.yaml")


def run_beta_sweep(model, columns, train_candidates, train_references):
    trip_ids = [
        trip_id
        for trip_id in sorted(train_candidates["trip_id"].unique())
        if trip_id not in BAD_TRAIN_TRIPS
    ][:5]
    if len(trip_ids) < 5:
        raise ValueError(f"Expected 5 usable training trips; found {len(trip_ids)}")

    rows = []
    for beta in BETA_VALUES:
        trip_scores = []
        for trip_id in trip_ids:
            trip, _, prediction = predict_trip(
                model, columns, train_candidates, trip_id, "train", beta=beta
            )
            score = top1(prediction, train_references[trip_id])
            trip_scores.append(score)
            rows.append({
                "record_type": "trip",
                "beta": beta,
                "trip_id": trip_id,
                "top1_accuracy": score,
            })
        rows.append({
            "record_type": "mean",
            "beta": beta,
            "trip_id": "",
            "top1_accuracy": float(np.mean(trip_scores)),
        })

    table = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "hmm_beta_sweep.csv", index=False)

    means = table.loc[table["record_type"] == "mean"]
    best_beta = int(means.sort_values(
        ["top1_accuracy", "beta"], ascending=[False, True]
    ).iloc[0]["beta"])
    print("\nBeta sweep mean top-1:")
    print(means[["beta", "top1_accuracy"]].to_string(index=False))
    print(f"Selected beta: {best_beta}")
    return best_beta


def run_student_test(model, columns, candidates, references, beta):
    rows = []
    for trip_id in list_trips("student_test"):
        trip, classifier_only, hmm_prediction = predict_trip(
            model, columns, candidates, trip_id, "student_test", beta=beta
        )
        reference = references[trip_id]

        classifier_top1 = top1(classifier_only, reference)
        hmm_top1 = top1(hmm_prediction, reference)

        classifier_continuity = route_continuity(classifier_only, trip)
        hmm_continuity = route_continuity(hmm_prediction, trip)

        rows.append({
            "record_type": "trip",
            "trip_id": trip_id,
            "classifier_only_top1": classifier_top1,
            "hmm_top1": hmm_top1,
            "classifier_only_route_continuity": classifier_continuity,
            "hmm_route_continuity": hmm_continuity,
        })

    table = pd.DataFrame(rows)
    metric_columns = [
        "classifier_only_top1",
        "hmm_top1",
        "classifier_only_route_continuity",
        "hmm_route_continuity",
    ]
    mean_row = {"record_type": "mean", "trip_id": ""}
    mean_row.update(table[metric_columns].mean().to_dict())
    table = pd.concat([table, pd.DataFrame([mean_row])], ignore_index=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "hmm_comparison.csv", index=False)
    print("\nStudent-test comparison:")
    print(table.to_string(index=False))

    worse = table.loc[
        (table["record_type"] == "trip")
        & (table["hmm_top1"] < table["classifier_only_top1"]),
        "trip_id",
    ].tolist()
    print("Trips where HMM top-1 got worse:", worse or "none")
    return table


def main():
    model, columns = load_model()
    train_candidates = load_candidate_table("train")
    test_candidates = load_candidate_table("student_test")
    train_references = load_reference_arcs("train")
    test_references = load_reference_arcs("student_test")

    best_beta = run_beta_sweep(
        model, columns, train_candidates, train_references
    )
    save_beta_to_config(best_beta)
    run_student_test(
        model, columns, test_candidates, test_references, best_beta
    )


if __name__ == "__main__":
    main()