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
TODO: exact command order (features -> train_compare -> optuna -> viterbi eval -> maps).

## AI use disclosure
TODO: list where AI tools were used.
