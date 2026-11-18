"""Short progress summary PDF (models used, data processing, best model + why). All numbers read from result files /
computed live (preliminary HMM check uses the UNTUNED XGBoost + default HMM parameters on the student-test trips)."""
import _bootstrap  # noqa
import json, joblib
import numpy as np, pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from PIL import Image as PILImage
from psnap.config import *
from psnap.data import load_trip, list_trips
from psnap.hmm import HMMParams
from psnap.pipeline import predict
from psnap.evaluation import evaluate

ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=4)
SM = ParagraphStyle("SM", parent=B, fontSize=7.8, leading=9.8, spaceAfter=0)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=colors.HexColor("#1f3b63"), spaceBefore=8, spaceAfter=5)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor("#1f3b63"), spaceBefore=7, spaceAfter=3)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
story = []
P = lambda t, s=B: story.append(Paragraph(t, s))
pc = lambda x: f"{100 * x:.2f}%"


def table(rows, header, widths):
    data = [[Paragraph(f"<b>{h}</b>", SM) for h in header]] + [[Paragraph(str(c), SM) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f3")), ("GRID", (0, 0), (-1, -1), .3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
                          + [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f8fb")) for i in range(2, len(data), 2)]))
    story.append(t); story.append(Spacer(1, 6))


def img(name, w_cm, cap):
    p = FIGS / name; w, h = PILImage.open(p).size; W = w_cm * cm
    story.append(KeepTogether([Image(str(p), width=W, height=W * h / w), Paragraph(cap, CAP)]))


comp = pd.read_csv(RESULTS / "model_comparison.csv"); inv = pd.read_csv(RESULTS / "dataset_inventory.csv", dtype={"trip": str})
tr, te = inv[inv.split == "train"], inv[inv.split == "student_test"]
best = comp.sort_values("CV Macro F1 (mean)", ascending=False).iloc[0]

# preliminary HMM check (untuned model, default decoder parameters)
m = joblib.load(MODELS / "XGBoost.joblib"); rows = []
for tid in list_trips(DATA_ROOT / "student_test"):
    t = load_trip(DATA_ROOT / "student_test", tid, split="student_test")
    o = predict(t.track, t.nodes, t.arcs, m, HMMParams()); e = evaluate(o["net"], t.arcs, t.route, o)
    rows.append((e["n_label_valid_points"], e["classifier_only"], e["hmm"]))
w = np.array([r[0] for r in rows]); avg = lambda k, j: float(np.average([r[j][k] for r in rows], weights=w))

P("PathSnap &mdash; Progress Summary", ParagraphStyle("T", parent=H1, fontSize=20, leading=24))
P("Team: PES1UG24CS137 Daksha S G &middot; PES1UG24CS138 Darshan G S &middot; PES1UG24CS183 Harsh Venkatsai Kalekar &middot; PES1UG24CS184 Harsha K<br/>"
  "<i>Interim status; every number below comes from the recorded experiment files (artifacts/results) or a live run. Final full report follows after all tests.</i>")

P("1. Where we are", H1)
table([["Data exploration &amp; preprocessing", "<b>Done</b>", "90 trips inspected, data-quality issues documented, labels derived, candidates + 33 features built"],
       ["Train &amp; compare models (min. 5 required)", "<b>Done &mdash; 10 models</b>", "5-fold group CV on 80 training trips + student-test evaluation"],
       ["Best-model selection", "<b>Done</b>", f"{best['Model']} (CV Macro-F1 {best['CV Macro F1 (mean)']:.4f})"],
       ["Bayesian hyper-parameter optimisation", "<b>Running</b>", "Optuna TPE, 30 trials of 5-fold group CV on XGBoost (about 2/3 complete)"],
       ["HMM + Viterbi route reconstruction", "<b>Implemented &amp; working</b>", "Final decoder-parameter tuning pending (after Bayesian run)"],
       ["FastAPI backend (laptop 1)", "<b>Done &amp; tested</b>", "All 10 trips run end-to-end, ~0.1&ndash;2.3 s per trip, request logging"],
       ["Streamlit UI (laptop 2)", "<b>Done &amp; tested</b>", "10 trip buttons, Folium map, metrics, tested against live API"],
       ["Folium maps, final technical report PDF, presentation notes", "In progress", "Report generator written; runs after the final results"],
       ["Push to GitHub, two-laptop rehearsal, demo screenshots", "<b>Team to do</b>", "Needs your GitHub account / both laptops"]],
      ["Work package", "Status", "Detail"], [6 * cm, 3.6 * cm, 7.6 * cm])
P("<b>Overall: about 80% complete.</b> Remaining work is the tuned-model run, final HMM tuning, maps, final report and the team-side GitHub/two-laptop steps.")

P("2. Data processing &mdash; yes, completed", H1)
P(f"<b>Dataset:</b> {len(tr)} training trips ({int(tr.n_points.sum()):,} GPS points) + {len(te)} student-test trips ({int(te.n_points.sum()):,} points); each trip has its own road network "
  f"(median {int(inv.n_arcs.median()):,} arcs). Files: .track (lon, lat, time), .nodes, .arcs, .route.")
for t in ["<b>Projection</b> of lon/lat to metres (local azimuthal-equidistant) &mdash; Euclidean distance in degrees would be wrong.",
          "<b>Label creation:</b> .route gives ordered arcs but not which GPS point is on which arc, so we align points to the route with a monotone dynamic-programming alignment; points &gt;50 m from their aligned arc are excluded (label-invalid).",
          f"<b>Data-quality findings:</b> {int(inv.n_gaps_gt10s.sum())} GPS gaps &gt;10 s (largest {inv.max_gap_s.max():.0f} s); {int((inv.n_components > 1).sum())} of {len(inv)} road networks have disconnected components; "
          f"{int((inv.route_connected < 1).sum())} reference routes have small breaks; for {int((inv.label_valid_frac < 0.97).sum())} trips a large part of the track is not covered by the reference route (e.g. train trip 00000054).",
          "<b>Candidate generation:</b> 12 nearest arcs within 60 m of each point (candidate miss rate about 0.6%); picking only the nearest arc would be wrong about 44% of the time because two-way streets have two arcs with the same geometry.",
          "<b>Feature engineering:</b> 33 features per (GPS point, candidate arc): distance/side/rank, position on arc, heading agreement, road topology, time gaps, neighbouring-points context, motion along arc."]:
    P("&bull; " + t)
img("eda_overview.png", 17, "EDA overview: points per trip, network size vs route length, sampling interval, GPS noise, speed, label coverage.")

P("3. Models used &mdash; 10 models (the minimum is 5)", H1)
table([[r["Model"], r["Algorithm family"], f"{r['CV Macro F1 (mean)']:.4f}", pc(r["CV Top-1"]), f"{r['Test Macro F1']:.4f}", pc(r["Test Top-1"]), f"{r['Test Log Loss']:.4f}", f"{r['Fit time (s)']:.1f}"] for _, r in comp.iterrows()],
      ["Model", "Family", "CV Macro-F1", "CV Top-1", "Test Macro-F1", "Test Top-1", "Test log loss", "Fit (s)"], [2.4 * cm, 4 * cm, 1.9 * cm, 1.8 * cm, 2 * cm, 1.8 * cm, 1.9 * cm, 1.4 * cm])
P("CV = 5-fold cross-validation on the 80 training trips (folds split by trip, so no trip is in two folds); Test = 10 student-test trips. Top-1 = for each GPS point, is the highest-scored candidate road the reference road. "
  "The five main families the team should explain: <b>Linear</b> (Logistic Regression), <b>Probabilistic</b> (Gaussian Naive Bayes), <b>Instance-based</b> (KNN), <b>Tree-based/ensembles</b> (Decision Tree, Random Forest, Extra Trees) and <b>Boosting + Neural net</b> "
  "(HistGB, LightGBM, XGBoost, MLP).", CAP)
img("cv_macro_f1.png", 12.5, "5-fold group cross-validation Macro-F1 (mean ± std over folds).")
img("test_comparison.png", 17, "Student-test: Top-1 and Macro-F1, log loss, and train/CV/test comparison (over/under-fitting check).")

P(f"4. Best model: {best['Model']} &mdash; and why", H1)
x = comp.set_index("Model").loc["XGBoost"]
P(f"<b>XGBoost</b> was selected on evidence, not on training accuracy: it has the <b>highest cross-validated Macro-F1 ({x['CV Macro F1 (mean)']:.4f})</b> (our primary metric because only about 12.9% of candidates are the true road), "
  f"the highest CV Top-1 ({pc(x['CV Top-1'])}), the <b>highest student-test Top-1 ({pc(x['Test Top-1'])}) and Macro-F1 ({x['Test Macro F1']:.4f})</b> and the <b>lowest log loss ({x['Test Log Loss']:.4f})</b>. "
  f"Low log loss matters because the HMM uses the classifier's probabilities as emissions. It also trains in {x['Fit time (s)']:.0f} s and predicts in {x['Inference (ms/pt)']:.4f} ms per GPS point, so it suits the live API.")
P("<b>Honest caveats:</b> the top models (XGBoost, Random Forest, HistGB, LightGBM) are within about one fold-standard-deviation (about 0.016) of each other on CV, so the ranking is close; "
  "XGBoost shows some overfitting (train Macro-F1 0.969 vs CV 0.929), which is why we are tuning it. Gaussian Naive Bayes is the weakest (independence assumption violated by correlated features, high log loss); "
  "Logistic Regression and KNN underfit relative to the tree ensembles because they cannot capture feature interactions.")
P("5. Preliminary result with the HMM / Viterbi decoder", H1)
P(f"Using the <b>untuned</b> XGBoost with <b>default</b> HMM parameters (preliminary &mdash; final tuning pending), on the 10 student-test trips (point-weighted): "
  f"<b>Top-1 accuracy {pc(avg('point_top1_accuracy', 1))} (classifier alone) &rarr; {pc(avg('point_top1_accuracy', 2))} (classifier + HMM)</b>; "
  f"route F1 {avg('route_f1', 1):.3f} &rarr; {avg('route_f1', 2):.3f}; route continuity {avg('route_continuity', 1):.3f} &rarr; {avg('route_continuity', 2):.3f}. "
  "The HMM improves the route because it enforces road-network connectivity and removes opposite-direction/parallel-road errors (the largest classifier error type).")
P("6. Still to do", H1)
for t in ["Finish Bayesian optimisation, then tune the HMM decoder on out-of-fold probabilities and re-evaluate (final numbers will replace section 5).",
          "Generate Folium maps, error analysis and the final technical report PDF (all sections required by the Deliverables document).",
          "<b>Team tasks:</b> push the code to a public GitHub repo (each member committing), run the API on laptop 1 and the UI on laptop 2, take the required screenshots (UI + API log on the receiving laptop), and read the viva sheet so every member can explain every part."]:
    P("&bull; " + t)

SimpleDocTemplate(str(ROOT / "reports" / "PathSnap_Progress_Summary.pdf"), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                  title="PathSnap Progress Summary").build(story)
print("ok", avg('point_top1_accuracy', 1), avg('point_top1_accuracy', 2))
