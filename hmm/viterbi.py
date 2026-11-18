"""Classifier-emission HMM and Viterbi route decoder."""
import networkx as nx
import numpy as np
import pandas as pd

from common.config import CFG

DEFAULT_BETA_M = 50.0
LOG_EPSILON = 1e-12


def _transition_scores(
    previous_rows: np.ndarray,
    current_rows: np.ndarray,
    candidates: pd.DataFrame,
    arc_u: np.ndarray,
    arc_v: np.ndarray,
    arc_lengths: np.ndarray,
    graph: nx.DiGraph,
    straight_distance: float,
    beta_m: float,
    distance_cache: dict[int, tuple[float, dict[int, float]]],
) -> np.ndarray:
    scores = np.full((len(previous_rows), len(current_rows)), -np.inf, dtype=float)
    cutoff = max(500.0, straight_distance * 4.0 + 2.0 * float(arc_lengths.max(initial=0)))
    current_arcs = candidates.loc[current_rows, "arc_id"].to_numpy(dtype=np.int64)
    current_starts = arc_u[current_arcs]

    for i, previous_row in enumerate(previous_rows):
        previous_arc = int(candidates.at[previous_row, "arc_id"])
        previous_fraction = float(candidates.at[previous_row, "proj_frac"])
        previous_end = int(arc_v[previous_arc])
        distances = {}
        if np.any(current_starts != previous_end):
            cached = distance_cache.get(previous_end)
            if cached is None or cached[0] < cutoff:
                distances = nx.single_source_dijkstra_path_length(
                    graph, previous_end, cutoff=cutoff, weight="weight"
                )
                if len(distance_cache) >= 256:
                    distance_cache.pop(next(iter(distance_cache)))
                distance_cache[previous_end] = (cutoff, distances)
            else:
                distances = cached[1]

        for j, current_row in enumerate(current_rows):
            current_arc = int(candidates.at[current_row, "arc_id"])
            current_fraction = float(candidates.at[current_row, "proj_frac"])
            if previous_arc == current_arc:
                network_distance = max(0.0, current_fraction - previous_fraction) * arc_lengths[previous_arc]
                if current_fraction + 0.1 < previous_fraction:
                    network_distance += 2.0 * beta_m
            else:
                current_start = int(arc_u[current_arc])
                middle_distance = 0.0 if previous_end == current_start else distances.get(current_start, np.inf)
                if not np.isfinite(middle_distance):
                    continue
                network_distance = (
                    (1.0 - previous_fraction) * arc_lengths[previous_arc]
                    + middle_distance
                    + current_fraction * arc_lengths[current_arc]
                )
            scores[i, j] = -abs(network_distance - straight_distance) / beta_m

    return scores


def decode(
    cand_df: pd.DataFrame,
    probs,
    trip: dict,
    beta_m: float | None = None,
    max_gap_s: float | None = None,
) -> list[int | None]:
    """Choose one candidate arc per observation; return None for gaps/no candidates.

    Disconnected candidate sets break the Viterbi chain and restart at the next
    observation rather than inventing a road-network connection.
    """
    from features.candidates import project_trip

    track = trip["track"]
    n_observations = len(track)
    decoded: list[int | None] = [None] * n_observations
    candidates = cand_df.reset_index(drop=True).copy()
    probabilities = np.asarray(probs, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(candidates):
        raise ValueError("probs must contain exactly one probability per candidate row")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Candidate probabilities must be finite values between 0 and 1")
    if candidates.empty:
        return decoded
    if not {"obs_idx", "arc_id"}.issubset(candidates.columns):
        raise ValueError("cand_df must contain obs_idx and arc_id columns")

    candidates["obs_idx"] = candidates["obs_idx"].astype(np.int64)
    if ((candidates["obs_idx"] < 0) | (candidates["obs_idx"] >= n_observations)).any():
        raise ValueError("Candidate obs_idx values must refer to rows in the trip track")
    if ((candidates["arc_id"] < 0) | (candidates["arc_id"] >= len(trip["arcs"]))).any():
        raise ValueError("Candidate arc_id values must refer to rows in the trip arcs")
    if "proj_frac" not in candidates:
        candidates["proj_frac"] = 0.5
    candidates["proj_frac"] = candidates["proj_frac"].astype(float).clip(0.0, 1.0)

    settings = CFG.get("hmm", {})
    beta_m = float(beta_m if beta_m is not None else settings.get("beta") or DEFAULT_BETA_M)
    max_gap_s = float(max_gap_s if max_gap_s is not None else settings.get("max_gap_s", 30))
    if beta_m <= 0 or max_gap_s <= 0:
        raise ValueError("beta_m and max_gap_s must be positive")

    projected = project_trip(trip)
    px, py = projected["px"], projected["py"]
    nodes, arcs = trip["nodes"], trip["arcs"]
    arc_u = arcs["u"].to_numpy(np.int64)
    arc_v = arcs["v"].to_numpy(np.int64)
    arc_lengths = np.hypot(
        projected["nx"][arc_v] - projected["nx"][arc_u],
        projected["ny"][arc_v] - projected["ny"][arc_u],
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.index)
    graph.add_weighted_edges_from(
        (int(u), int(v), float(length))
        for u, v, length in zip(arc_u, arc_v, arc_lengths)
    )

    rows_by_observation = {
        int(obs_idx): group.index.to_numpy(dtype=np.int64)
        for obs_idx, group in candidates.groupby("obs_idx", sort=True)
    }
    emissions = np.log(np.clip(probabilities, LOG_EPSILON, 1.0))
    timestamps = pd.to_numeric(track["t"], errors="coerce").to_numpy(float)
    segment_observations: list[int] = []
    segment_rows: list[np.ndarray] = []
    segment_scores: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []
    distance_cache: dict[int, tuple[float, dict[int, float]]] = {}

    def finish_segment() -> None:
        if not segment_observations:
            return
        state = int(np.argmax(segment_scores[-1]))
        selected = [state]
        for backpointer in reversed(backpointers):
            state = int(backpointer[state])
            selected.append(state)
        selected.reverse()
        for obs_idx, rows, state_idx in zip(segment_observations, segment_rows, selected):
            decoded[obs_idx] = int(candidates.at[rows[state_idx], "arc_id"])

    for obs_idx in range(n_observations):
        current_rows = rows_by_observation.get(obs_idx)
        if current_rows is None:
            finish_segment()
            segment_observations.clear()
            segment_rows.clear()
            segment_scores.clear()
            backpointers.clear()
            continue

        if segment_observations:
            previous_obs = segment_observations[-1]
            elapsed = timestamps[obs_idx] - timestamps[previous_obs]
            if not np.isfinite(elapsed) or elapsed <= 0 or elapsed > max_gap_s:
                finish_segment()
                segment_observations.clear()
                segment_rows.clear()
                segment_scores.clear()
                backpointers.clear()

        if not segment_observations:
            segment_observations.append(obs_idx)
            segment_rows.append(current_rows)
            segment_scores.append(emissions[current_rows])
            continue

        previous_obs = segment_observations[-1]
        straight_distance = float(np.hypot(px[obs_idx] - px[previous_obs], py[obs_idx] - py[previous_obs]))
        transition = _transition_scores(
            segment_rows[-1],
            current_rows,
            candidates,
            arc_u,
            arc_v,
            arc_lengths,
            graph,
            straight_distance,
            beta_m,
            distance_cache,
        )
        total_scores = segment_scores[-1][:, None] + transition
        best_previous = np.argmax(total_scores, axis=0)
        best_scores = total_scores[best_previous, np.arange(len(current_rows))]
        if not np.isfinite(best_scores).any():
            finish_segment()
            segment_observations.clear()
            segment_rows.clear()
            segment_scores.clear()
            backpointers.clear()
            segment_observations.append(obs_idx)
            segment_rows.append(current_rows)
            segment_scores.append(emissions[current_rows])
            continue

        segment_observations.append(obs_idx)
        segment_rows.append(current_rows)
        segment_scores.append(emissions[current_rows] + best_scores)
        backpointers.append(best_previous)

    finish_segment()
    return decoded
