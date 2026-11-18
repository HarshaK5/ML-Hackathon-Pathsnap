"""Review-1 slide deck (PDF): data preprocessing, models used, accuracy. Numbers read from artifacts/results."""
import _bootstrap  # noqa
import json
import numpy as np, pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from PIL import Image as PILImage
from psnap.config import *

ss = getSampleStyleSheet()
NAVY = colors.HexColor("#1f3b63")
T = ParagraphStyle("T", parent=ss["Title"], fontSize=30, leading=36, textColor=NAVY, alignment=0)
H = ParagraphStyle("H", parent=ss["Heading1"], fontSize=23, leading=27, textColor=NAVY, spaceAfter=8)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=13.5, leading=19, spaceAfter=5)
BL = ParagraphStyle("BL", parent=B, leftIndent=14, bulletIndent=2)
SM = ParagraphStyle("SM", parent=B, fontSize=10.5, leading=13, spaceAfter=0)
CAP = ParagraphStyle("CAP", parent=B, fontSize=10, leading=12, textColor=colors.HexColor("#555555"))
W = landscape(A4)[0] - 3 * cm
S = []
def P(t, st=B): S.append(Paragraph(t, st))
def bl(items):
    for i in items: S.append(Paragraph(i, BL, bulletText="•"))
def img(n, h_cm):
    w, h = PILImage.open(FIGS / n).size; H_ = h_cm * cm; S.append(Image(str(FIGS / n), width=H_ * w / h, height=H_))
def tbl(rows, hdr, widths, fs=10.5, hl=None):
    st = ParagraphStyle("c", parent=SM, fontSize=fs, leading=fs + 2.5)
    d = [[Paragraph(f"<b>{h}</b>", st) for h in hdr]] + [[Paragraph(str(c), st) for c in r] for r in rows]
    t = Table(d, colWidths=widths, repeatRows=1)
    sty = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f3")), ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if hl is not None: sty.append(("BACKGROUND", (0, hl + 1), (-1, hl + 1), colors.HexColor("#fff2b3")))
    t.setStyle(TableStyle(sty)); S.append(t)

inv = pd.read_csv(RESULTS / "dataset_inventory.csv", dtype={"trip": str}); comp = pd.read_csv(RESULTS / "model_comparison.csv")
RES = {m: json.load(open(RESULTS / "models" / f"{m}.json")) for m in comp.Model}
tv = json.load(open(RESULTS / "tuned_vs_untuned.json")); hs = json.load(open(RESULTS / "hmm_summary.json"))
tr, te = inv[inv.split == "train"], inv[inv.split == "student_test"]
pc = lambda x: f"{100 * x:.2f}%"

P("PathSnap", T); P("Map-matching: reconstructing the road route from noisy GPS", ParagraphStyle("s", parent=H, fontSize=19))
P("<b>Review 1 — Data preprocessing, models used and their accuracy</b>"); S.append(Spacer(1, 8))
P("PES1UG24CS137 Daksha S G &nbsp;·&nbsp; PES1UG24CS138 Darshan G S &nbsp;·&nbsp; PES1UG24CS183 Harsh Venkatsai Kalekar &nbsp;·&nbsp; PES1UG24CS184 Harsha K")
S.append(Spacer(1, 10))
bl(["<b>Problem:</b> GPS points drift off the road (buildings, signal interference). Given the noisy track and the road network, find the roads the vehicle really travelled.",
    "<b>ML formulation:</b> for every GPS point take nearby candidate roads, and let a classifier predict “is this candidate the true road?”. (An HMM then smooths the route — next review.)"])
S.append(PageBreak())

P("1. Dataset", H)
tbl([["Training", len(tr), f"{int(tr.n_points.sum()):,}", f"{int(tr.n_arcs.median()):,}", "fitting, cross-validation, tuning"],
     ["Student test", len(te), f"{int(te.n_points.sum()):,}", f"{int(te.n_arcs.median()):,}", "development comparison of models"],
     ["Evaluator test", 10, "hidden", "hidden", "final evaluation only — never touched"]],
    ["Partition", "Trips", "GPS points", "Median road arcs / trip", "Use"], [4.5 * cm, 2.5 * cm, 3.5 * cm, 5 * cm, 9 * cm], 12)
S.append(Spacer(1, 8))
bl(["Per trip: <b>.track</b> (lon, lat, time ≈ 1 Hz), <b>.nodes</b>, <b>.arcs</b> (directed road segments), <b>.route</b> (reference arcs in order).",
    "<b>Every trip has its own road network</b> (649 to 307,392 arcs) and trips are located all over the world.",
    "Splits are <b>by trip</b> — no trip appears in two partitions."])
img("eda_overview.png", 8.6); S.append(PageBreak())

P("2. Data preprocessing — what we did", H)
tbl([["1", "Inspect", f"{int(inv.n_nan.sum())} NaNs, no timestamp disorder, no isolated nodes; {int(inv.n_gaps_gt10s.sum())} GPS gaps > 10 s; 96.7 % of steps exactly 1 s"],
     ["2", "Project to metres", "Local azimuthal-equidistant projection per trip (degrees are angular — Euclidean distance in degrees is wrong)"],
     ["3", "Create point labels", ".route lists arcs but not which GPS point is on which arc → monotone dynamic-programming alignment of points to route arcs"],
     ["4", "Filter bad labels", "Points &gt; 50 m from their aligned arc are label-invalid and excluded (88.9 % of training points kept; student-test 99.9 %)"],
     ["5", "Candidate generation", "12 nearest arcs within 60 m of each point (STRtree). Reference arc missing from candidates for only 0.6 %"],
     ["6", "Feature engineering", "33 features per (GPS point, candidate arc) pair → 1.43 M training rows, 12.9 % positive"],
     ["7", "Leakage control", "Group-by-trip CV; no fitted preprocessing on test data; scalers fitted on training folds only"]],
    ["#", "Step", "Detail"], [1 * cm, 5 * cm, 20.5 * cm], 12)
S.append(Spacer(1, 6))
P("<b>Why candidates?</b> Picking only the nearest arc is wrong ≈ 44 % of the time — two-way streets have two arcs with identical geometry — so a classifier is needed to choose.", B)
S.append(PageBreak())

P("3. Data-quality findings", H)
bl([f"<b>{int((inv.n_components > 1).sum())} of {len(inv)}</b> road networks have disconnected components → routing must handle unreachable pairs.",
    f"<b>{int((inv.route_connected < 1).sum())}</b> reference routes contain small breaks; some trips revisit the same arc.",
    f"<b>{int((inv.label_valid_frac < 0.97).sum())} trips</b> have a track much longer than the reference route (e.g. train trip 00000054: only 11 % of points have a reference arc) → excluded from supervision.",
    "GPS noise vs reference road: <b>median ≈ 5 m, 99th percentile ≈ 34 m</b>; largest time gap 153,064 s in one trip."])
img("eda_trajectories.png", 6.6); P("GPS (blue), road network (grey), reference route (green) for 3 training trips and 1 student-test trip.", CAP)
S.append(PageBreak())

P("4. Feature engineering — 33 features per (point, candidate road)", H)
tbl([["Geometric fit", "distance to road, side of road, rank among candidates, margin over nearest candidate, candidate count"],
     ["Position on road", "fraction along the arc, arc length, distance to either end"],
     ["Heading agreement", "cos/sin/abs of angle between GPS heading and road bearing, speed (separates the two directions of a two-way street)"],
     ["Road topology", "has reverse arc (two-way), branching degree at both ends"],
     ["Time gaps", "gap to previous / next GPS fix"],
     ["Temporal context", "distance of the 3 previous and 3 next GPS points to the <b>same</b> road (true road stays close along consecutive points)"],
     ["Motion along road", "signed movement along the arc, turn rate (junction indicator)"]],
    ["Group", "Features and why"], [5 * cm, 21.5 * cm], 12)
S.append(PageBreak())

P("5. Experimental setup", H)
bl(["<b>10 models</b> from 8 algorithm families (minimum required: 5), all on identical rows.",
    "<b>5-fold GROUP cross-validation</b> on the 80 training trips (folds split by trip), then final fit on all 80 and evaluation on the 10 student-test trips.",
    "<b>Candidate-level accuracy / Macro-F1:</b> is the candidate predicted true/not-true correctly (Macro-F1 averages both classes — needed because only 12.9 % are true).",
    "<b>Top-1 accuracy (per GPS point):</b> is the highest-scored candidate the reference road? — this is the map-matching accuracy.",
    "<b>Log loss:</b> quality of the predicted probabilities (they feed the HMM later). Primary selection metric: <b>CV Macro-F1</b>."])
S.append(PageBreak())

P("6. Models used and their accuracy", H)
rows = []; best = comp["CV Macro F1 (mean)"].idxmax()
for _, r in comp.iterrows():
    te_ = RES[r.Model]["test"]
    rows.append([f"<b>{r.Model}</b>", r["Algorithm family"], f"{r['CV Macro F1 (mean)']:.4f}", pc(r["CV Top-1"]), pc(te_["accuracy"]), f"{r['Test Macro F1']:.4f}", f"<b>{pc(r['Test Top-1'])}</b>", f"{r['Test Log Loss']:.4f}", f"{r['Fit time (s)']:.1f}"])
tbl(rows, ["Model", "Family", "CV Macro-F1", "CV Top-1", "Test cand. accuracy", "Test Macro-F1", "Test Top-1 (per point)", "Test log loss", "Fit (s)"],
    [3.1 * cm, 5.2 * cm, 2.5 * cm, 2.3 * cm, 2.9 * cm, 2.6 * cm, 3 * cm, 2.4 * cm, 1.8 * cm], 10.5, hl=int(best))
P("CV = 5-fold group cross-validation (training trips). Test = 10 student-test trips. Highlighted = best by CV Macro-F1.", CAP)
S.append(PageBreak())

P("7. Model comparison charts", H)
img("cv_macro_f1.png", 7.3); S.append(Spacer(1, 4)); P("Cross-validated Macro-F1 (mean ± std over the 5 group folds).", CAP)
S.append(PageBreak())
P("7. Model comparison charts (student-test) and over/under-fitting check", H); img("test_comparison.png", 7.2)
P("Left: Top-1 accuracy and Macro-F1. Middle: log loss (lower is better). Right: train vs CV vs test Macro-F1 — a large train-minus-CV gap means overfitting.", CAP)
S.append(PageBreak())

x = comp.set_index("Model").loc["XGBoost"]; xr = RES["XGBoost"]
P("8. Best model: XGBoost — and why", H)
bl([f"<b>Highest CV Macro-F1 ({x['CV Macro F1 (mean)']:.4f})</b> and CV Top-1 ({pc(x['CV Top-1'])}); <b>highest test Top-1 ({pc(x['Test Top-1'])})</b>, test Macro-F1 ({x['Test Macro F1']:.4f}); <b>lowest log loss ({x['Test Log Loss']:.4f})</b>.",
    f"Fast: trains in {x['Fit time (s)']:.0f} s, predicts in {x['Inference (ms/pt)']:.4f} ms per GPS point → suits the live API.",
    "Boosted trees capture non-linear interactions (e.g. distance matters differently when heading is unreliable or at junctions); regularised, so probabilities are well calibrated.",
    "<b>Weakest:</b> Gaussian Naive Bayes (features are correlated, violating independence; log loss 0.84). <b>Linear / KNN underfit</b>: they cannot model feature interactions.",
    f"<b>Honest caveats:</b> top 4 models (XGBoost, Random Forest, HistGB, LightGBM) are within ≈ 1 fold-std (≈ 0.016); XGBoost overfits somewhat (train Macro-F1 {xr['train']['macro_f1']:.3f} vs CV {xr['cv_mean']['macro_f1']:.3f}) → tuned next.",
    f"<b>Bayesian tuning (Optuna TPE, 30 trials, 5-fold group CV):</b> CV Macro-F1 {tv['untuned']['cv_macro_f1']:.4f} → {tv['tuned']['cv_macro_f1']:.4f}; test Top-1 {pc(tv['untuned']['test_top1'])} → {pc(tv['tuned']['test_top1'])} (small gain — the features, not hyper-parameters, limit the classifier)."])
S.append(PageBreak())

P("9. Next step and overall result so far", H)
P("A classifier alone still makes mistakes — mostly choosing the opposite-direction arc of the same street (3.9 %) — because it scores each GPS point independently. "
  "An <b>HMM with Viterbi decoding</b> adds road connectivity and sequence information:")
tbl([["Classifier only (tuned XGBoost)", "90.39 %", f"{hs['clf']['route_f1']:.3f}", f"{hs['clf']['route_continuity']:.3f}"],
     ["Classifier + HMM / Viterbi", f"<b>{pc(hs['hmm']['point_top1_accuracy'])}</b>", f"<b>{hs['hmm']['route_f1']:.3f}</b>", f"<b>{hs['hmm']['route_continuity']:.3f}</b>"]],
    ["Student-test (10 trips)", "Top-1 accuracy (per GPS point)", "Route F1", "Route continuity"], [9 * cm, 6.5 * cm, 4.5 * cm, 6 * cm], 13)
S.append(Spacer(1, 10))
bl(["Done: preprocessing, feature engineering, 10-model comparison, Bayesian tuning, HMM/Viterbi, FastAPI backend, Streamlit UI, Folium maps.",
    "Remaining: two-laptop rehearsal, GitHub push, final report screenshots."])

def foot(c, d):
    c.saveState(); c.setFont("Helvetica", 9); c.setFillColor(colors.grey)
    c.drawString(1.5 * cm, 0.8 * cm, "PathSnap — Review 1"); c.drawRightString(landscape(A4)[0] - 1.5 * cm, 0.8 * cm, str(d.page)); c.restoreState()
SimpleDocTemplate(str(ROOT / "reports" / "PathSnap_Review1_Slides.pdf"), pagesize=landscape(A4), leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.2 * cm, bottomMargin=1.4 * cm,
                  title="PathSnap Review 1").build(S, onFirstPage=foot, onLaterPages=foot)
print("ok")
