"""Folium maps of HMM predictions for two student_test trips: the one with the biggest
HMM top-1 gain and the one with the smallest gain (or a loss), from results/hmm_comparison.csv.
Predictions use hmm.evaluate.predict_trip (artifacts/model.joblib + hmm.viterbi.decode with
hmm.beta from config.yaml), i.e. exactly what hmm_comparison.csv measured.
Map layers come from maps.folium_maps.make_trip_map; this script adds a legend.
Output: maps/out/<trip>_predicted.html
Run: python -m maps.make_prediction_maps
"""
import pandas as pd

from common.config import CFG, REPO_ROOT
from hmm.evaluate import load_candidate_table, load_model, predict_trip
from maps.folium_maps import make_trip_map

OUT = REPO_ROOT / "maps" / "out"

LEGEND = """
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
            padding: 10px 12px; border: 1px solid #999; border-radius: 4px;
            font: 13px/1.5 Arial, sans-serif; color: #222;">
  <b>{title}</b><br>
  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#1E88E5;"></span>
  &nbsp;GPS observations<br>
  <span style="display:inline-block;width:18px;height:3px;background:#9E9E9E;vertical-align:middle;"></span>
  &nbsp;Road network (toggle in layer control)<br>
  <span style="display:inline-block;width:18px;height:5px;background:#2E7D32;vertical-align:middle;"></span>
  &nbsp;Reference route<br>
  <span style="display:inline-block;width:18px;height:5px;background:#D81B60;vertical-align:middle;"></span>
  &nbsp;Predicted route (classifier + HMM)<br>
  <span style="color:#555;">{subtitle}</span>
</div>
"""


def pick_trips() -> list[tuple[str, pd.Series]]:
    t = pd.read_csv(REPO_ROOT / "results" / "hmm_comparison.csv", dtype={"trip_id": str})
    t = t[t["record_type"] == "trip"].assign(gain=lambda d: d["hmm_top1"] - d["classifier_only_top1"])
    best, worst = t.loc[t["gain"].idxmax()], t.loc[t["gain"].idxmin()]
    return [("biggest HMM gain", best), ("smallest HMM gain", worst)]


def main() -> None:
    beta = CFG["hmm"]["beta"]
    model, columns = load_model()
    candidates = load_candidate_table("student_test")
    for why, row in pick_trips():
        tid = row["trip_id"]
        _, _, hmm_arcs = predict_trip(model, columns, candidates, tid, "student_test", beta=beta)
        plain = OUT / f"{tid}.html"
        existing = plain.read_bytes() if plain.exists() else None
        path = make_trip_map(tid, "student_test", predicted_arcs=hmm_arcs)
        html = path.read_text(encoding="utf-8")
        legend = LEGEND.format(
            title=f"student_test trip {tid} ({why})",
            subtitle=(f"top-1 classifier-only {row['classifier_only_top1']:.3f} -> "
                      f"HMM {row['hmm_top1']:.3f} (beta={beta})"))
        html = html.replace("</body>", legend + "</body>", 1)
        out = OUT / f"{tid}_predicted.html"
        out.write_text(html, encoding="utf-8")
        if existing is None:
            plain.unlink()
        else:
            plain.write_bytes(existing)
        print(f"{why}: trip {tid}, gain {row['gain']:+.4f} -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
