import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from common.config import REPO_ROOT
from data_processing.align_labels import align_trip
from data_processing.loader import list_trips, load_trip
from features.build_features import FEATURE_COLS, build_trip_features
from hmm.viterbi import decode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap API")

MODEL: Any = None
MODEL_FILE: Path | None = None
FEATURE_COLUMNS: list[str] | None = None


def load_model() -> None:
    """Prefer the tuned artifact; use the baseline LightGBM if needed."""
    global MODEL, MODEL_FILE, FEATURE_COLUMNS

    artifacts = REPO_ROOT / "artifacts"
    path = next(
        (
            candidate
            for candidate in (artifacts / "model.joblib", artifacts / "lgbm.joblib")
            if candidate.is_file()
        ),
        None,
    )
    if path is None:
        raise FileNotFoundError(
            "Missing artifacts/model.joblib and artifacts/lgbm.joblib"
        )

    saved = joblib.load(path)
    if isinstance(saved, dict) and "model" in saved:
        MODEL = saved["model"]
        FEATURE_COLUMNS = saved.get("feature_cols")
    else:
        MODEL = saved
        FEATURE_COLUMNS = None

    columns_path = artifacts / "feature_cols.json"
    if not FEATURE_COLUMNS and columns_path.is_file():
        with columns_path.open(encoding="utf-8") as file:
            columns = json.load(file)
        FEATURE_COLUMNS = (
            columns.get("feature_cols", columns.get("features"))
            if isinstance(columns, dict)
            else columns
        )

    if not FEATURE_COLUMNS:
        FEATURE_COLUMNS = list(FEATURE_COLS)

    MODEL_FILE = path
    log.info("Loaded model file: %s", path.name)


@app.on_event("startup")
def load_model_on_startup() -> None:
    try:
        load_model()
    except Exception:
        # Keep /health available if the model file is missing or unreadable.
        log.exception("Could not load the model at startup")


def get_model() -> tuple[Any, list[str]]:
    if MODEL is None or FEATURE_COLUMNS is None:
        try:
            load_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc

    return MODEL, FEATURE_COLUMNS


def positive_probabilities(model: Any, values: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(values), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    positive_column = np.flatnonzero(classes == 1)

    if raw.ndim != 2 or len(positive_column) != 1:
        raise ValueError("Model did not return probabilities for class 1.")

    return raw[:, positive_column[0]]


def best_candidate_per_point(candidates, probabilities, point_count: int):
    predictions: list[int | None] = [None] * point_count
    if candidates.empty:
        return predictions

    scored = candidates[["obs_idx", "arc_id"]].copy()
    scored["_probability"] = probabilities

    for obs_idx, group in scored.groupby("obs_idx", sort=False):
        best_index = group["_probability"].idxmax()
        predictions[int(obs_idx)] = int(scored.at[best_index, "arc_id"])

    return predictions


def top1_accuracy(predictions, reference_arcs) -> float:
    """Count missing predictions as incorrect."""
    if not reference_arcs:
        return 0.0

    correct = [
        prediction is not None and int(prediction) == int(reference)
        for prediction, reference in zip(predictions, reference_arcs)
    ]
    return float(np.mean(correct))


def make_predicted_path(predictions, trip) -> list[list[float]]:
    """Convert consecutive unique predicted arcs to [latitude, longitude] nodes."""
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    path: list[list[float]] = []
    previous_arc = None

    for arc_id in predictions:
        if arc_id is None:
            continue
        arc_id = int(arc_id)
        if arc_id == previous_arc:
            continue
        previous_arc = arc_id

        for node_id in (int(arcs.at[arc_id, "u"]), int(arcs.at[arc_id, "v"])):
            point = [
                float(nodes.at[node_id, "lat"]),
                float(nodes.at[node_id, "lon"]),
            ]
            if not path or point != path[-1]:
                path.append(point)

    return path


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_FILE.name if MODEL_FILE is not None else None,
    }


@app.get("/trips")
def trips():
    try:
        return {"trips": list_trips("eval")}
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/{trip_id}")
def predict(trip_id: str):
    try:
        available = list_trips("eval")
    except (RuntimeError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if trip_id not in available:
        raise HTTPException(status_code=404, detail="Unknown trip ID")

    inference_start = time.perf_counter()

    try:
        # No reference route is loaded during the prediction stage.
        trip = load_trip(trip_id, "eval", with_route=False)
        candidates = build_trip_features(trip)
        model, columns = get_model()

        if candidates.empty:
            probabilities = np.asarray([], dtype=float)
            classifier_arcs = [None] * len(trip["track"])
            hmm_arcs = [None] * len(trip["track"])
        else:
            missing = [column for column in columns if column not in candidates.columns]
            if missing:
                raise ValueError(f"Features missing from candidate table: {missing}")

            probabilities = positive_probabilities(
                model, candidates[columns].to_numpy(dtype=np.float32)
            )
            classifier_arcs = best_candidate_per_point(
                candidates, probabilities, len(trip["track"])
            )

            # Viterbi currently expects proj_frac; features provide f_proj_frac.
            decode_candidates = candidates.copy()
            if "proj_frac" not in decode_candidates.columns:
                decode_candidates["proj_frac"] = decode_candidates["f_proj_frac"]

            hmm_arcs = decode(decode_candidates, probabilities, trip)

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Prediction failed for trip %s", trip_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    # Stop the inference timer before loading or aligning the reference route.
    inference_ms = (time.perf_counter() - inference_start) * 1000

    try:
        trip_with_route = load_trip(trip_id, "eval", with_route=True)
        aligned = align_trip(trip_with_route)
        reference_arcs = aligned["true_arc_id"].tolist()
    except Exception as exc:
        log.exception("Could not calculate metrics for trip %s", trip_id)
        raise HTTPException(status_code=500, detail="Could not calculate trip metrics") from exc

    try:
        from hmm.metrics import route_continuity

        continuity = route_continuity(hmm_arcs, trip_with_route)
    except ImportError:
        continuity = None

    metrics = {
        "top1_accuracy": top1_accuracy(hmm_arcs, reference_arcs),
        "route_continuity": continuity,
        "classifier_only_top1": top1_accuracy(classifier_arcs, reference_arcs),
    }

    gps = trip["track"][["lat", "lon"]].astype(float).values.tolist()
    log.info(
        "trip=%s points=%d top1=%.4f continuity=%s ms=%.1f",
        trip_id,
        len(gps),
        metrics["top1_accuracy"],
        "null" if continuity is None else f"{continuity:.4f}",
        inference_ms,
    )

    return {
        "trip_id": trip_id,
        "gps": gps,
        "predicted_arcs": hmm_arcs,
        "predicted_path": make_predicted_path(hmm_arcs, trip),
        "metrics": metrics,
        "inference_ms": round(inference_ms, 3),
    }