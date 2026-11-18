import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from features.build_features import FEATURE_COLS
from hmm.viterbi import decode
from models.lgbm import fit_model, predict_proba
from models.common_eval import evaluate_model


def make_trip():
    nodes = pd.DataFrame(
        {
            "lon": [10.000, 10.001, 10.002, 10.003],
            "lat": [50.000, 50.000, 50.000, 50.000],
        }
    )
    arcs = pd.DataFrame({"u": [0, 1, 2, 1], "v": [1, 2, 3, 0]})
    track = pd.DataFrame(
        {
            "lon": [10.0002, 10.0012, 10.0022],
            "lat": [50.000, 50.000, 50.000],
            "t": [0.0, 1.0, 2.0],
        }
    )
    return {"trip_id": "test", "nodes": nodes, "arcs": arcs, "track": track}


class LightGBMTests(unittest.TestCase):
    def test_fit_and_predict_returns_positive_probability_per_row(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(80, len(FEATURE_COLS))).astype(np.float32)
        labels = (x[:, 0] < 0).astype(np.int8)
        frame = pd.DataFrame(x, columns=FEATURE_COLS)
        frame["label"] = labels

        model = fit_model(frame, n_estimators=20, min_child_samples=2)
        probabilities = predict_proba(model, frame)

        self.assertEqual(probabilities.shape, (len(frame),))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_estimator_runs_through_shared_grouped_evaluator(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(64, len(FEATURE_COLS))).astype(np.float32)
        frame = pd.DataFrame(x, columns=FEATURE_COLS)
        labels = (x[:, 0] < 0).astype(np.int8)
        groups = np.repeat(["trip_a", "trip_b", "trip_c", "trip_d"], 16)
        observation_ids = np.tile(np.repeat(np.arange(8), 2), 4)

        with TemporaryDirectory() as output_dir:
            results = evaluate_model(
                estimator=fit_model(frame.assign(label=labels), n_estimators=20, min_child_samples=2),
                X=frame,
                y=labels,
                groups=groups,
                observation_ids=observation_ids,
                model_name="lgbm_test",
                n_splits=2,
                results_dir=output_dir,
            )

        self.assertEqual(results["record_type"].tolist(), ["fold", "fold", "mean", "std"])
        self.assertTrue(np.isfinite(results.loc[0:1, "macro_f1"]).all())


class ViterbiTests(unittest.TestCase):
    def test_decoder_prefers_connected_route_over_local_probability_peak(self):
        trip = make_trip()
        candidates = pd.DataFrame(
            {
                "obs_idx": [0, 0, 1, 2],
                "arc_id": [0, 3, 1, 2],
                "proj_frac": [0.2, 0.5, 0.2, 0.2],
            }
        )

        route = decode(candidates, np.array([0.55, 0.9, 0.99, 0.99]), trip)

        self.assertEqual(route, [0, 1, 2])

    def test_decoder_marks_missing_candidates_and_restarts_after_time_gap(self):
        trip = make_trip()
        trip["track"].loc[2, "t"] = 60.0
        candidates = pd.DataFrame(
            {"obs_idx": [0, 2], "arc_id": [0, 2], "proj_frac": [0.2, 0.2]}
        )

        route = decode(candidates, np.array([0.9, 0.8]), trip, max_gap_s=30)

        self.assertEqual(route, [0, None, 2])

    def test_decoder_returns_none_when_no_candidates_exist(self):
        trip = make_trip()
        candidates = pd.DataFrame(columns=["obs_idx", "arc_id"])

        route = decode(candidates, np.array([]), trip)

        self.assertEqual(route, [None, None, None])


if __name__ == "__main__":
    unittest.main()
