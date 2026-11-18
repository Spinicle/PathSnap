# PathSnap: map-matching and route reconstruction

UE24CS352A Machine Learning hackathon, PES University.

Noisy GPS points -> candidate road segments -> classifier scores -> HMM/Viterbi -> continuous route.
The FastAPI service runs on Laptop 1 and the UI on Laptop 2.

## Structure
| folder | contents | owner |
|---|---|---|
| `data_processing/` | loading, projection, EDA, label alignment | A |
| `features/` | candidate generation, feature table | A |
| `models/` | 5-model comparison, metrics | B |
| `optimisation/` | Optuna (Bayesian) search | B |
| `hmm/` | HMM + Viterbi decoder | C |
| `maps/` | Folium maps | C |
| `api/` | FastAPI backend | D |
| `ui/` | Streamlit evaluator UI | D |
| `results/` | CSVs, logs, figures (all report numbers come from here) | all |
| `artifacts/` | trained model + configs | B |
| `docs/CONTRACTS.md` | shared interfaces | all |

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
Data is not committed. By default it is read from `../Student folder` (see `config.yaml`),
or set `PATHSNAP_DATA`. Quick check: `python -m data_processing.loader`

## Run
```bash
# Laptop 1
set PATHSNAP_EVAL_DATA=<path to evaluator trips>
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Laptop 2
set PATHSNAP_API=http://<laptop1-ip>:8000
streamlit run ui/app.py
```
Allow port 8000 through the Windows firewall on Laptop 1.

## Reproduce results
Run from the repo root, in this order. Seed 42 everywhere.
```bash
set PATHSNAP_DATA=../Student folder

# 1. candidate generation + feature table -> cache/features_*.parquet, cache/labels_*.parquet
python -m features.build_features --split all

# 2. models: GroupKFold(5) CV on train + student_test -> results/runs/<model>.csv,
#    results/runs/<model>_student_test.csv, artifacts/<model>.joblib
python -m models.logreg
python -m models.nb
python -m models.rf_test
python -m models.lgbm
python -m models.mlp

# 3. comparison table + plots -> results/model_comparison_final.csv, results/figures/
python -m models.compare

# 4. Optuna (TPE) on LightGBM -> results/optuna_trials.csv, artifacts/best_params.json,
#    artifacts/model.joblib, results/tuning_comparison.csv
python -m optimisation.optuna_search --trials 40 --timeout 1200
#    --trials 40 is the cap; 17 trials completed within the 1200 s timeout (see results/optuna_trials.csv;
#    the first 10 are TPE's random startup trials). Tuned CV macro F1 0.9043 vs 0.9038 untuned;
#    student-test metrics within 0.003 of untuned; inference ~8x slower (475 trees / 211 leaves vs 300 / 31).
#    artifacts/model.joblib (tuned) is what the API serves.

# 5. HMM / Viterbi evaluation -> results/hmm_beta_sweep.csv, results/hmm_comparison.csv, hmm.beta in config.yaml
python -m hmm.evaluate
#    Beta sweep on train trips 00000001-00000005 over {10, 25, 50, 100, 200}; the chosen beta is
#    written to (and read from) hmm.beta in config.yaml.
#    Top-1 is averaged differently in the two tables: hmm_comparison.csv's mean row is the unweighted
#    mean of the 10 per-trip values (classifier-only 0.8721, HMM 0.9158), while
#    model_comparison_final.csv is pooled over all 19,160 student_test points (classifier-only 0.8545;
#    HMM pooled 0.9047). Same points, same true arcs; long low-scoring trips weigh more when pooled.

# 6. Folium maps -> maps/out/
python -m maps.folium_maps
python -m maps.make_prediction_maps   # HMM predictions, biggest/smallest HMM gain trips

# report figures -> results/figures/
python -m report.make_figures

# 7. API (Laptop 1)
set PATHSNAP_EVAL_DATA=<path to evaluator trips>
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 8. UI (Laptop 2)
set PATHSNAP_API=http://<laptop1-ip>:8000
streamlit run ui/app.py
```
Random Forest: `models/rf_test.py` is the run the results come from. It reuses the
hyperparameters of `models/rf.py` (the original CV-only exploration run, not a required step)
with `n_jobs=-1`. With `random_state=42` fixed, `n_jobs` only changes speed, not results.
`artifacts/random_forest.joblib` (138 MB) is not committed (too large); regenerate it with
`python -m models.rf_test`.

## AI use disclosure
- Claude Code: label alignment, candidate generation, feature builder, baselines, comparison script (reviewed and explained by Yadu)
- TODO: teammates add their entries
