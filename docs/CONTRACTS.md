# Interfaces (agree on these first, change only together)

## Feature table (features/build_features.py -> models/)
One row per (GPS point, candidate arc).

| column | type | notes |
|---|---|---|
| trip_id | str | group key for GroupKFold |
| obs_idx | int | row index in .track |
| arc_id | int | candidate arc |
| f_* | float | feature columns, prefixed `f_` |
| label | int | 1 if arc_id is the aligned true arc (training/dev only) |

## Model artifact (models/ -> hmm/, api/)
`artifacts/model.joblib` + `artifacts/feature_cols.json`.
`predict_proba(df[feature_cols]) -> P(label=1)` per row.

## Decoder (hmm/)
`decode(cand_df, probs, trip) -> list[arc_id or None]` (one per GPS point).

## API
`GET /trips` -> `{"trips": ["00000015", ...]}`

`POST /predict/{trip_id}` ->
```json
{
  "trip_id": "...",
  "gps": [[lat, lon], ...],
  "predicted_arcs": [123, 124, ...],
  "predicted_path": [[lat, lon], ...],
  "metrics": {"top1_accuracy": 0.0, "route_continuity": 0.0, "classifier_only_top1": 0.0},
  "inference_ms": 0
}
```
The reference route is never sent to the UI.
