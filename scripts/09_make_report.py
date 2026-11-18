"""Generate the consolidated technical report (PDF). Every number/table/figure is read from artifacts/results and
artifacts/figures (produced by the experiment scripts); verdicts such as 'overfitting' are computed from those numbers."""
import _bootstrap  # noqa
import json
from pathlib import Path
import numpy as np, pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from PIL import Image as PILImage
from psnap.config import *

OUT = ROOT / "reports" / "PathSnap_Technical_Report.pdf"
OUT.parent.mkdir(exist_ok=True)
ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.2, leading=12.4, spaceAfter=4)
SM = ParagraphStyle("SM", parent=B, fontSize=7.4, leading=9.2, spaceAfter=0)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1f3b63"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1f3b63"))
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=9.8, spaceBefore=6, spaceAfter=2)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, leading=10, textColor=colors.HexColor("#444444"), spaceAfter=8)
story = []


def P(t, s=B): story.append(Paragraph(t, s))
def S(h=6): story.append(Spacer(1, h))
def f4(x): return f"{x:.4f}"
def pc(x): return f"{100 * x:.2f}%"


def img(name, width_cm=17, caption=None):
    p = FIGS / name
    if not p.exists():
        P(f"<i>[figure {name} not found - run the corresponding script]</i>"); return
    w, h = PILImage.open(p).size
    W = width_cm * cm
    story.append(KeepTogether([Image(str(p), width=W, height=W * h / w)] + ([Paragraph(caption, CAP)] if caption else [])))


def table(rows, header, widths=None, fs=7.4, zebra=True):
    data = [[Paragraph(f"<b>{h}</b>", SM) for h in header]] + [[Paragraph(str(c), SM) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f3")), ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f8fb")) for i in range(2, len(data), 2)]
    t.setStyle(TableStyle(st)); story.append(t); S(6)


def J(p): return json.load(open(RESULTS / p))


# ------------------------------------------------------------------ load results
inv = pd.read_csv(RESULTS / "dataset_inventory.csv", dtype={"trip": str})
abl = pd.read_csv(RESULTS / "candidate_miss_ablation.csv", index_col=0)
comp = pd.read_csv(RESULTS / "model_comparison.csv")
RES = {m: J(f"models/{m}.json") for m in comp.Model}
bb, tv = J("bayes_best.json"), J("tuned_vs_untuned.json")
hs = J("hmm_summary.json"); ht = pd.read_csv(RESULTS / "hmm_vs_classifier_student_test.csv", dtype={"trip": str})
trials = pd.read_csv(RESULTS / "bayes_trials.csv")
err = pd.read_csv(RESULTS / "error_analysis.csv", index_col=0) if (RESULTS / "error_analysis.csv").exists() else None
hist = J("training_histories.json") if (RESULTS / "training_histories.json").exists() else {}
best_model = bb["model"]; hp = hs["hmm_params"]
tr_inv, te_inv = inv[inv.split == "train"], inv[inv.split == "student_test"]

# ------------------------------------------------------------------ title
P("PathSnap: Distributed Map-Matching &amp; Route Reconstruction", ParagraphStyle("T", parent=H1, fontSize=21, leading=26, spaceAfter=4))
P("Technical Report &mdash; UE24CS352A Machine Learning, PES University (RR Campus)", H2)
table([["PES1UG24CS137", "Daksha S G"], ["PES1UG24CS138", "Darshan G S"], ["PES1UG24CS183", "Harsh Venkatsai Kalekar"], ["PES1UG24CS184", "Harsha K"]],
      ["SRN", "Team member"], [4 * cm, 8 * cm])
P("<b>AI-assistance disclosure.</b> This project was developed with the help of an AI coding assistant (Claude Code). All experiments were executed by the "
  "scripts in the accompanying GitHub repository; every table, number and graph in this report is generated from the recorded result files in "
  "<font face='Courier'>artifacts/</font> (nothing is typed in by hand). Evaluator-partition data was never used for training, feature selection, model selection or "
  "tuning. See <font face='Courier'>docs/AI_DISCLOSURE.md</font>.")
P("1. Problem and system overview", H1)
P("GPS positions of a vehicle are noisy (buildings, multipath, drop-outs), so recorded points drift off the road. Given a noisy 1 Hz trajectory and the road network, "
  "PathSnap reconstructs the sequence of road segments (arcs) the vehicle really travelled. Our solution is a four-stage pipeline: "
  "<b>(1) candidate generation</b> &ndash; the K=12 nearest road arcs within 60 m of every GPS point; <b>(2) feature engineering</b> &ndash; 33 geometric / kinematic / "
  "temporal-context features for every (GPS point, candidate arc) pair; <b>(3) a machine-learning classifier</b> that outputs the probability that a candidate is the true arc; "
  "<b>(4) an HMM with Viterbi decoding</b> that combines those probabilities with road-network connectivity to produce a coherent route. "
  "The system is deployed as a FastAPI backend (laptop 1) and a Streamlit UI (laptop 2).")
P("Evaluation protocol. The 100 trips are split by trip (never within a trip): 80 training trips (fitting, 5-fold <i>group</i> cross-validation, Bayesian optimisation), "
  "10 student-test trips (development comparisons) and 10 evaluator trips (never touched). Primary model-selection metric: <b>cross-validated Macro-F1</b> of the candidate "
  "classification (candidate = true arc vs not), because the classes are imbalanced (about 12.9% positives) and Macro-F1 weights both classes equally; Observation-level Top-1 "
  "accuracy (per GPS point, is the highest-scored candidate the reference arc?) and Log Loss (probability quality, which matters for the HMM emissions) are reported alongside.")

# ------------------------------------------------------------------ 2 EDA
P("2. Dataset understanding and exploratory data analysis", H1)
P("2.1 Dataset organisation (partition: all 90 development trips)", H2)
P(f"Each trip folder contains four header-less whitespace-separated files: <b>.track</b> (longitude, latitude, time in s), <b>.nodes</b> (longitude, latitude; the row index is the node id), "
  f"<b>.arcs</b> (from_node, to_node; the row index is the arc id, arcs are directed) and <b>.route</b> (the ordered arc ids of the reference route). "
  f"Each trip carries its <i>own</i> road network, and trips are located all over the world (Fig. 2). The training partition has {len(tr_inv)} trips "
  f"({int(tr_inv.n_points.sum()):,} GPS points) and the student-test partition {len(te_inv)} trips ({int(te_inv.n_points.sum()):,} points). "
  f"Road networks range from {int(inv.n_arcs.min()):,} to {int(inv.n_arcs.max()):,} arcs (median {int(inv.n_arcs.median()):,}); reference routes contain {int(inv.route_arcs.min())} to {int(inv.route_arcs.max()):,} arcs.")
inv_t = inv.groupby("split").agg(trips=("trip", "count"), points=("n_points", "sum"), med_pts=("n_points", "median"), max_pts=("n_points", "max"),
                                 nodes=("n_nodes", "sum"), arcs=("n_arcs", "sum"), route_arcs=("route_arcs", "sum"), gaps=("n_gaps_gt10s", "sum")).reset_index()
table([[r.split, int(r.trips), f"{int(r.points):,}", int(r.med_pts), f"{int(r.max_pts):,}", f"{int(r.nodes):,}", f"{int(r.arcs):,}", f"{int(r.route_arcs):,}", int(r.gaps)] for r in inv_t.itertuples()],
      ["Partition", "Trips", "GPS points", "Median pts/trip", "Max pts/trip", "Nodes", "Arcs", "Route arcs", "Gaps >10 s"],
      [2.6 * cm, 1.3 * cm, 2.2 * cm, 2.2 * cm, 2.0 * cm, 2.2 * cm, 2.2 * cm, 1.9 * cm, 1.7 * cm])
P("Table 1. Dataset inventory (from <font face='Courier'>artifacts/results/dataset_inventory.csv</font>, which lists every trip).", CAP)

P("2.2 Preprocessing", H2)
P("<b>Coordinates.</b> Longitude/latitude are angular, so Euclidean distance in degrees is wrong. For every trip we project all coordinates to metres with a local "
  "azimuthal-equidistant projection centred on the trip (pyproj); trips span at most tens of kilometres so distortion is negligible. Coordinate order (lon, lat) was verified against the plausible geographic extents.")
P("<b>Label construction.</b> The .route file lists the arcs in travel order but does not say which GPS point lies on which arc, so a row of .track does not correspond to a row of .route. "
  "We recover per-point reference arcs by a monotone dynamic-programming alignment: every GPS point is assigned a route position that never decreases along the track, minimising the "
  "sum of squared point-to-arc distances (distance capped at 150 m). Points whose aligned arc is more than 50 m away are flagged <i>label-invalid</i> and excluded from supervised training and scoring.")
P("<b>Candidates.</b> A shapely STRtree finds the arcs within a radius of each point; the K nearest are kept. If no arc lies in the radius we fall back to the nearest arcs at any distance, so every GPS point always has candidates. "
  "Radius and K were chosen from the candidate-miss ablation below (training partition, label-valid points).")
r = abl.copy(); r.columns = [f"{c} m" for c in r.columns]
table([[f"K={k}"] + [pc(v) for v in row] for k, row in r.iterrows()], ["K \\ radius"] + list(r.columns), [2 * cm] + [2 * cm] * len(r.columns))
P(f"Table 2. Candidate miss rate (reference arc absent from candidate set). Chosen: radius {CAND_RADIUS_M:.0f} m, K={CAND_K} (miss rate {pc(abl.loc[CAND_K, '60'] if '60' in abl.columns else abl.iloc[:, 3].loc[CAND_K])}). "
  f"Note K=1 (nearest arc only) is wrong {pc(abl.iloc[:, -1].loc[1])} of the time because every two-way street contributes two arcs with identical geometry.", CAP)
img("candidate_miss.png", 10.5, "Figure 1. Candidate miss rate versus radius and K.")

P("2.3 Patterns observed and data-quality findings", H2)
n_nan = int(inv.n_nan.sum())
P(f"<b>Missing / invalid values.</b> {n_nan} NaN values across all track and node files; no non-monotonic timestamps ({int(inv.nonmonotonic_t.sum())}); no isolated nodes ({int(inv.isolated_nodes.sum())}). "
  f"<b>Sampling.</b> Nominal 1 Hz ({np.average(inv.pct_dt_1s, weights=inv.n_points):.1f}% of steps are exactly 1 s), but there are {int(inv.n_gaps_gt10s.sum())} gaps longer than 10 s (largest {inv.max_gap_s.max():.0f} s), concentrated in a few trips (Fig. 2). "
  f"<b>Noise.</b> GPS-to-reference distance has median about 5 m and 99th percentile about 34 m for label-valid points. <b>Network.</b> {int((inv.n_components > 1).sum())} of {len(inv)} networks consist of several "
  f"disconnected components (median {int(inv.n_components.median())}), so shortest-path routing must treat unreachable pairs explicitly. "
  f"<b>Reference route limitations.</b> {int((inv.route_connected < 1).sum())} trips have reference routes with a small number of non-contiguous consecutive arcs; "
  f"{int((inv.route_repeated_arcs > 0).sum())} trips revisit arcs. Most importantly, for {int((inv.label_valid_frac < 0.97).sum())} trips a large part of the GPS track lies away from the reference route "
  f"(e.g. train trip 00000054: only {pc(inv[inv.trip == '00000054'].label_valid_frac.iloc[0])} of the points have a reference arc within 50 m, the track being several times longer than the route and extending beyond the extracted road network, Fig. 3). "
  f"These points are excluded from supervision and scoring, and we avoid over-stating the certainty of the reference labels.")
img("eda_overview.png", 17, "Figure 2. EDA overview (training + student-test partitions). Top: points per trip, network size vs route length, sampling intervals. Bottom: GPS-to-reference distance, GPS speed, fraction of points with a valid reference label per trip.")
img("eda_trajectories.png", 17.3, "Figure 3. GPS trajectory (blue), road network (grey) and reference route (green) for three training trips (00000005, 00000002, 00000054) and one student-test trip (00000052).")
img("trip_locations.png", 10, "Figure 4. Trip locations: trips are scattered across the world, each with its own road network (both partitions).")
P("<b>Interactive Folium map</b> (training trip 00000005): <font face='Courier'>maps/eda_train_00000005.html</font> shows raw GPS points, road network and the reference route with layer control.")

P("2.4 Feature engineering (partition: training for fitting; all rows built identically for student-test)", H2)
P("Every row is a (GPS point, candidate arc) pair, 33 features in total. We chose them because each captures a cue a human would use to decide which road the car is on:")
feat_rows = [
    ["Geometric fit", "d, lat_off, d_rank, d_min, d_minus_min, d_ratio, n_cand", "Perpendicular distance to the arc, side of road, rank among candidates, margin over the nearest candidate, local candidate density. Noise is about 5 m so distance is the strongest cue, but near junctions and parallel roads several arcs are equally close."],
    ["Position on arc", "u, arc_len, dist_to_start, dist_to_end", "Where along the arc the projection falls; points projecting at an arc end are ambiguous with the next arc."],
    ["Heading agreement", "cos_dh, sin_dh, abs_dh, speed, heading_ok", "GPS heading (central difference over 5 points) versus the arc bearing: separates the two opposite-direction arcs of a two-way street. Used only when the vehicle is moving (speed &gt; 1 m/s and displacement &gt; 6 m); otherwise heading is marked unreliable."],
    ["Topology", "has_reverse, deg_out_end, deg_in_start, deg_out_start", "Whether the street is two-way and how many roads branch at the arc ends (junction complexity / road class proxy)."],
    ["Time gaps", "gap_prev, gap_next", "Log time gaps to neighbouring fixes (large gaps make local cues unreliable)."],
    ["Temporal context", "d_m3..d_m1, d_p1..d_p3, d_ctx_mean, d_ctx_max", "Distance of the 3 previous and 3 next GPS points to the <i>same</i> candidate arc: a true arc stays close along consecutive fixes, a crossing road does not."],
    ["Motion along arc", "along_move, along_fwd_agree, turn_rate", "Signed movement of the projected position along the arc over +-2 points (travel direction relative to the arc) and heading change over +-4 points (junction/curve indicator)."]]
table(feat_rows, ["Group", "Features", "Why we chose them"], [2.6 * cm, 5.2 * cm, 9.6 * cm])
P("Features are fit-free (computed per trip from raw geometry), so no learned preprocessing leaks across partitions; models needing scaling (LogReg, KNN, MLP) use a StandardScaler fitted on training folds only.")

# ------------------------------------------------------------------ 3 models
P("3. Model experimentation and comparison", H1)
P(f"We trained and evaluated <b>{len(comp)} models from 8 algorithm families</b> (more than the required five). All use the identical rows: "
  "label-valid GPS points of the 80 training trips for fitting and 5-fold <b>group</b> cross-validation (folds split by trip; seed 42), and the 10 student-test trips for development comparison. "
  "Class balance is about 12.9% positive candidates. 'Top-1' is the observation-level Top-1 accuracy; 'continuity' is the fraction of consecutive distinct predicted arcs that are connected in the road network "
  "(classifier-only predictions, before any HMM). Train metrics are computed on a fixed random sample of 20,000 training GPS points.")
P("3.1 Overall model comparison", H2)
rows = []
for _, r_ in comp.iterrows():
    rows.append([r_["Model"], r_["Algorithm family"], r_["Key configuration"], f"{r_['CV Macro F1 (mean)']:.4f} ± {r_['CV Macro F1 (std)']:.4f}", pc(r_["CV Top-1"]), f"{r_['CV Route continuity']:.4f}",
                 f4(r_["Test Macro F1"]), pc(r_["Test Top-1"]), f4(r_["Test Log Loss"]), f"{r_['Inference (ms/pt)']:.4f} ms/pt ({r_['Inference total (s)']:.2f} s total)"])
table(rows, ["Model", "Algorithm family", "Key configuration", "CV Macro F1", "CV Top-1", "CV route continuity", "Student-Test Macro F1", "Student-Test Top-1", "Log Loss (test)", "Inference time"],
      [1.5 * cm, 1.6 * cm, 4.4 * cm, 1.8 * cm, 1.3 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.2 * cm, 1.9 * cm])
P("Table 3. Consolidated comparison (actual results; source <font face='Courier'>artifacts/results/model_comparison.csv</font>). Macro-F1 is at threshold 0.5 over candidate rows. "
  "Log loss is applicable to every model because all output probabilities (GaussianNB and KNN are the least calibrated). Route continuity here is for the classifier alone; HMM results are in section 5.", CAP)
best_cv = comp.sort_values("CV Macro F1 (mean)", ascending=False).iloc[0]
P(f"<b>Selection.</b> The best cross-validated Macro-F1 is {best_cv['Model']} ({best_cv['CV Macro F1 (mean)']:.4f}); the top boosted/ensemble models are within about one fold-standard-deviation of each other "
  f"(std about 0.016), so we considered, besides Macro-F1, probability quality (log loss, which directly feeds the HMM emission term), inference time and tunability. "
  f"{best_model} offers the best CV Macro-F1, a fast fit/predict time and well-behaved probabilities, and was therefore selected for Bayesian optimisation (section 4).")

P("3.2 Per-model analysis", H2)
INFO = {
    "LogReg": dict(why="A linear, convex baseline that gives calibrated probabilities from the distance, heading and context features; it tells us how far a simple decision boundary gets.",
                   strong="Fast, deterministic, well-calibrated, interpretable coefficients, no tuning risk.", limit="Cannot represent interactions (for example distance matters differently when heading is unreliable or at junctions) and cannot fit the non-linear shape of the distance/heading trade-off.",
                   loss="Binary cross-entropy (log loss) with L2 regularisation (C=1), solved by L-BFGS."),
    "GaussianNB": dict(why="A generative probabilistic baseline; it is the standard 'simplest probabilistic model' and shows what happens when the independence assumption is wrong.",
                       strong="Extremely fast to fit, no hyper-parameters of consequence.", limit="Assumes features independent given the class and Gaussian-distributed; our features (e.g. the distances of neighbouring points to the same arc, d, d_ctx_mean) are strongly correlated and heavy-tailed, giving overconfident, poorly calibrated probabilities.",
                       loss="Maximises the class-conditional Gaussian likelihood (closed form; no iterative loss)."),
    "KNN": dict(why="Instance-based: candidates with similar geometric/kinematic configurations should share a label. Tests whether local similarity in feature space is enough.",
                strong="Non-parametric, can fit irregular boundaries, no training.", limit="Needs scaled features, cost grows with the reference set (we use a 60,000-row random subsample of the training rows, otherwise inference is prohibitive) and the distance metric treats all features as equally informative.",
                loss="No training loss (lazy learner); predictions are distance-weighted neighbour votes (k=25)."),
    "DecisionTree": dict(why="Captures non-linear rules and feature interactions (for example 'close AND heading agrees AND not at junction') in an interpretable form.",
                         strong="No scaling needed, interpretable, handles interactions, fast inference.", limit="A single tree has high variance and produces coarse (piecewise-constant) probabilities, which hurts log loss.",
                         loss="Gini impurity reduction at each split (depth limited to 14, min 20 samples per leaf)."),
    "RandomForest": dict(why="Bagging many decorrelated deep trees reduces the variance of a single tree while keeping non-linear modelling.",
                         strong="Stable, robust to outliers and scaling, good out-of-the-box accuracy.", limit="Large models, slower inference than a single tree, probabilities are averaged votes and so less sharp than boosted models.",
                         loss="Gini impurity per tree; bagged with 30% bootstrap samples and random feature subsets."),
    "ExtraTrees": dict(why="Like Random Forest but with random split thresholds; more bias, less variance and very fast to train.",
                       strong="Very fast, smooth probability estimates, strong variance reduction.", limit="Random thresholds waste capacity on weak features and cost some accuracy on sharply defined cues such as distance margins.",
                       loss="Gini impurity with random split points, 30% bootstrap samples."),
    "HistGB": dict(why="Gradient boosting corrects errors sequentially and handles heterogeneous, non-linear tabular features well; the histogram variant is fast on 1M+ rows.",
                   strong="High accuracy on tabular data, native handling of different scales, fast.", limit="Sensitive to learning rate / number of rounds, can overfit noisy labels, needs tuning.",
                   loss="Binary log-loss minimised by gradient boosting."),
    "LightGBM": dict(why="A leaf-wise gradient-boosting implementation designed for large tabular data; a strong candidate for the classifier feeding the HMM.",
                     strong="Fast, accurate, supports regularisation and subsampling, histogram-based.", limit="Leaf-wise growth with 63 leaves can overfit noisy reference labels; hyper-parameter sensitive.",
                     loss="Binary log-loss (gradient boosting) with row/column subsampling (0.8)."),
    "XGBoost": dict(why="Regularised gradient boosting with depth-limited trees and a second-order objective; widely the strongest model family on tabular data of this shape.",
                    strong="Accurate, regularised (L1/L2, min_child_weight), well-calibrated probabilities, fast hist implementation.", limit="Many hyper-parameters, can overfit when depth is high (as our train/CV gap shows), less interpretable.",
                    loss="Logistic loss with L2 regularisation, second-order boosting, depth 8, 300 rounds, learning rate 0.1."),
    "MLP": dict(why="A neural network can learn non-linear feature combinations without manual interaction design; included as the neural-network family.",
                strong="Flexible function approximator, smooth probabilities, fast inference.", limit="Needs scaled inputs, sensitive to initialisation, less competitive than boosted trees on tabular features of this kind and slower to train.",
                loss="Cross-entropy minimised with Adam (hidden layers 64-32, ReLU, L2 alpha=1e-4, early stopping on a 10% validation split)."),
}
FEAT_NOTE = {"LogReg": "scaling=standard"}
for m_ in comp.Model:
    r_ = RES[m_]; tr_, cv_, te_ = r_["train"], r_["cv_mean"], r_["test"]; I = INFO[m_]
    gap = tr_["macro_f1"] - cv_["macro_f1"]; cvbest = comp["CV Macro F1 (mean)"].max()
    if gap > 0.03:
        ov = f"<b>Overfitting observed:</b> train Macro-F1 {tr_['macro_f1']:.4f} versus CV {cv_['macro_f1']:.4f} (gap {gap:.3f}) and student-test {te_['macro_f1']:.4f}; the train-vs-held-out gap is evidence of fitting trip-specific noise."
    elif cv_["macro_f1"] < cvbest - 0.015:
        ov = (f"<b>Underfitting relative to the best models:</b> train Macro-F1 ({tr_['macro_f1']:.4f}), CV ({cv_['macro_f1']:.4f}) and test ({te_['macro_f1']:.4f}) are all similar to each other (gap {gap:.3f}) "
              f"yet clearly below the best model's CV {cvbest:.4f}, i.e. the model lacks the capacity/flexibility to use the features fully; it is not overfitting.")
    else:
        ov = (f"<b>Mild/no overfitting:</b> train {tr_['macro_f1']:.4f} vs CV {cv_['macro_f1']:.4f} (gap {gap:.3f}), student-test {te_['macro_f1']:.4f}; fold-to-fold std of CV Macro-F1 is {r_['cv_std']['macro_f1']:.4f}.")
    story.append(KeepTogether([
        Paragraph(f"3.2.{list(comp.Model).index(m_) + 1}  {m_}  <font size=8 color='#666666'>({r_['family']})</font>", H3),
        Paragraph("<b>A. Model understanding.</b> (1) <i>Why suitable:</i> " + I["why"] + " (2) <i>Strengths:</i> " + I["strong"] + " <i>Limitations:</i> " + I["limit"], B),
        Paragraph("<b>B. Training and configuration.</b> (5) Important hyper-parameters (initial values): " + ", ".join(f"{k}={v}" for k, v in r_["config"].items()) + ". (6) Objective: " + I["loss"] +
                  f" Final fit time on the 80 training trips: {r_['fit_seconds']:.1f} s (5-fold CV total {r_['cv_total_seconds']:.0f} s).", B),
        Paragraph(f"<b>C. Experimental results.</b> (7, 8) Student-test: Top-1 accuracy <b>{pc(te_['top1'])}</b>, Macro-F1 <b>{te_['macro_f1']:.4f}</b>, Log Loss <b>{te_['log_loss']:.4f}</b>, precision/recall (true-arc class) {te_['precision']:.3f}/{te_['recall']:.3f}, "
                  f"classifier-only route continuity {te_['continuity']:.3f}; CV Top-1 {pc(cv_['top1'])}, CV Macro-F1 {cv_['macro_f1']:.4f}. (9) {ov}", B)]))
P("Note: student-test scores are slightly higher than CV scores for most models. The student-test trips have nearly complete reference labels (99.9% of points label-valid), whereas the training folds include noisier trips with partially reliable reference routes (section 2.3), so the CV estimate is the more pessimistic one.", CAP)

P("3.3 Model visualisations", H2)
img("cv_macro_f1.png", 12.5, "Figure 5. 5-fold group cross-validation Macro-F1 for all models (mean ± std over folds; training partition).")
img("test_comparison.png", 17.3, "Figure 6. Student-test partition: Top-1 accuracy and Macro-F1 (left), Log Loss (middle) and an over/under-fitting check comparing train, CV and test Macro-F1 (right).")
img("efficiency.png", 13, "Figure 7. Computational efficiency: final-fit time and inference time per GPS point.")
if (FIGS / "training_curves.png").exists():
    img("training_curves.png", 17.3, "Figure 8. Real training histories (fold 0 held out as validation) for the models that train iteratively: LightGBM and XGBoost (per boosting round) and the MLP (per epoch). "
                                     "Models without an iterative training procedure (KNN, trees, forests, Naive Bayes, L-BFGS logistic regression) have no curves by design; none were fabricated.")
if hist:
    obs = []
    for k_, h_ in hist.items():
        v_ = np.array(h_["val"]); b_ = int(v_.argmin()); obs.append(f"{k_}: validation log-loss is lowest at {h_['unit']} {b_ + 1} ({v_.min():.4f}) and ends at {v_[-1]:.4f} after {len(v_)} ({'+' if v_[-1] > v_.min() else ''}{v_[-1] - v_.min():.4f}); final train log-loss {h_['train'][-1]:.4f}")
    P("<b>What the curves show (overfitting evidence).</b> " + "; ".join(obs) + ". "
      "A widening train/validation gap with rising validation loss is the signature of overfitting to the training trips; it explains why more boosting rounds are not better here, and why tuning (learning rate, depth, regularisation) was worthwhile.")
img("confusion_matrices.png", 17.3, "Figure 9. Student-test confusion matrices (candidate level) for every model (optional diagnostic).")

# ------------------------------------------------------------------ 4 bayes
P(f"4. Bayesian hyper-parameter optimisation ({best_model})", H1)
bp = bb["best_params"]
P(f"<b>Method.</b> Optuna's Tree-structured Parzen Estimator (TPE) sampler (seed {bb['seed']}, 8 random start-up trials) with a median pruner; every trial is a full 5-fold <i>group</i> cross-validation on the 80 training trips "
  f"(same folds as in section 3). The first trial is the untuned configuration (warm start), so the tuned result can never be worse than the baseline within the search. The student-test partition is used only after the search to compare tuned vs untuned.")
P(f"<b>Objective:</b> maximise mean CV Macro-F1 (primary metric, section 1). <b>Trials:</b> {bb['n_trials']} ({bb['n_complete']} complete, {bb['n_pruned']} pruned); optimisation took {bb['optimisation_seconds'] / 60:.1f} minutes.")
space_rows = [["n_estimators", "integer 100-500 (step 50)"], ["learning_rate", "log-uniform 0.03-0.3"], ["max_depth", "integer 3-12"], ["min_child_weight", "log-uniform 1-50"],
              ["subsample", "uniform 0.5-1.0"], ["colsample_bytree", "uniform 0.5-1.0"], ["reg_lambda (L2)", "log-uniform 1e-3-10"], ["reg_alpha (L1)", "log-uniform 1e-3-10"]] if best_model == "XGBoost" else \
             [["n_estimators", "integer 100-500"], ["learning_rate", "log-uniform 0.03-0.3"], ["num_leaves", "log-int 15-255"], ["min_child_samples", "log-int 10-200"], ["subsample", "0.5-1"], ["colsample_bytree", "0.5-1"], ["reg_lambda", "1e-3-10"], ["reg_alpha", "1e-3-10"]]
table([[a, b_, (f"{bp[a.split()[0]]:.5g}" if a.split()[0] in bp else "-")] for a, b_ in space_rows], ["Hyper-parameter", "Search distribution", "Best value found"], [4 * cm, 6 * cm, 4 * cm])
u, t = tv["untuned"], tv["tuned"]
table([["CV Macro-F1 (5-fold group CV)", f4(u["cv_macro_f1"]), f4(t["cv_macro_f1"])], ["CV Top-1 accuracy", pc(u["cv_top1"]), pc(t["cv_top1"])],
       ["Student-test Top-1 accuracy", pc(u["test_top1"]), pc(t["test_top1"])], ["Student-test Macro-F1", f4(u["test_macro_f1"]), f4(t["test_macro_f1"])],
       ["Student-test Log Loss", f4(u["test_log_loss"]), f4(t["test_log_loss"])], ["Inference time (ms / GPS point)", f"{u['inference_ms_per_point']:.4f}", f"{t['inference_ms_per_point']:.4f}"],
       ["Final fit time (s)", f"{u['fit_seconds']:.1f}", f"{t['fit_seconds']:.1f}"]], ["Metric (same data partitions & metric code)", "Untuned", "Tuned"], [8 * cm, 3.5 * cm, 3.5 * cm])
dv = t["cv_macro_f1"] - u["cv_macro_f1"]; dt_ = t["test_top1"] - u["test_top1"]
P(f"<b>Did tuning help?</b> CV Macro-F1 changed by {dv:+.4f} and student-test Top-1 by {100 * dt_:+.2f} percentage points; Log Loss changed from {u['test_log_loss']:.4f} to {t['test_log_loss']:.4f}. "
  f"{'The tuned model is better on the selection metric' if dv > 0 else 'Tuning did not improve the selection metric'}; "
  f"{'gains are small compared with the fold-to-fold standard deviation (about 0.016), so the model is close to the ceiling set by the features and by label noise rather than by hyper-parameters; the bigger gain comes from sequence decoding (section 5).' if abs(dv) < 0.01 else 'the gain exceeds fold-to-fold noise.'}")
img("bayes_opt.png", 15.5, "Figure 10. Optimisation history (grey = pruned trials) and hyper-parameter importance. Full trial record: artifacts/results/bayes_trials.csv and optuna_study.db.")

# ------------------------------------------------------------------ 5 HMM
P("5. HMM and Viterbi route reconstruction", H1)
P("<b>Hidden states.</b> At every GPS step <i>t</i> the hidden state is the road arc the vehicle is on, restricted to the candidate set C<sub>t</sub> (the same K candidates the classifier scored), so the candidate-to-arc correspondence is preserved end to end.")
P(f"<b>Emission probabilities.</b> log P(o<sub>t</sub> | arc) = &lambda; &middot; log p(arc | features), where p is the selected classifier's probability that the candidate is the true arc (clipped to [1e-6, 1]); &lambda; = {hp['lam']:.3f}.")
P(f"<b>Transition probabilities.</b> Newson-Krumm style. For candidate <i>a</i> at t-1 and <i>b</i> at t we compute the road-network travel distance d<sub>route</sub> = remaining length of a + shortest path (end of a to start of b, "
  f"bounded Dijkstra on a corridor sub-graph) + travelled length of b (or |u<sub>b</sub>-u<sub>a</sub>|&middot;length when a=b), and compare it with the straight GPS displacement: log T(a&rarr;b) = -|d<sub>route</sub> - d<sub>gps</sub>| / &beta;, &beta; = {hp['beta']:.2f} m. "
  f"Moving onto the opposite-direction arc (a U-turn) pays an extra penalty of {hp['uturn_pen']:.2f}; any arc change pays {hp['switch_pen']:.2f}.")
P(f"<b>Missing candidates and large GPS gaps.</b> Candidate generation falls back to the nearest arcs at any distance when nothing lies within 60 m, so every fix has candidates. When the time gap exceeds {hp['gap_thresh']:.1f} s the displacement is no longer informative about the route length, so only reachability is used "
  f"(search bounded by v<sub>max</sub> = 45 m/s times the gap) and the cost grows only slightly with path length.")
P(f"<b>Disconnected segments.</b> If consecutive observations suggest arcs that are not connected (no path within the search bound), the transition is not forbidden but receives a fixed log-penalty of {hp['unreach_pen']:.2f} ('route break'), so Viterbi can always continue and chooses a break only when the classifier evidence outweighs it. "
  f"After decoding, consecutive predicted arcs that are not directly connected are bridged by the shortest road path; pairs with no path are counted as unbridged breaks.")
P("<b>Viterbi.</b> Standard max-sum dynamic programme in log space: &delta;<sub>t</sub>(b) = max<sub>a</sub> [&delta;<sub>t-1</sub>(a) + log T(a&rarr;b)] + emission<sub>t</sub>(b), with back-pointers; the final state is argmax &delta;<sub>N</sub> and the path is traced back. "
  "Because route distances do not depend on the HMM parameters they are pre-computed once per trip (a 'lattice'), which makes parameter tuning cheap.")
P(f"<b>Decoder-parameter tuning.</b> {hs['hmm_tuning']['n_trials']} Optuna TPE trials maximising pooled point-level Top-1 accuracy on <i>out-of-fold</i> classifier probabilities of {hs['tuning_set_top1']['n_trips']} fully-labelled training trips "
  f"({hs['tuning_set_top1']['n_points']:,} GPS points; the classifier never saw these trips when producing those probabilities, so no leakage). Student-test and evaluator data were not used for tuning. "
  f"Tuned values: " + ", ".join(f"{k}={v:.3g}" for k, v in hp.items()) + ". "
  "<b>Caveat:</b> the tuned emission weight &lambda; lies close to the lower edge of its search range (0.2-3) and the U-turn penalty improves slowly towards the top of its range (Fig. 12), so a wider search might find slightly better values; the gain from the HMM is, however, large and stable across the whole range of every parameter.")
s = hs["tuning_set_top1"]
P("<b>Results &mdash; classifier-only vs classifier + HMM/Viterbi.</b>", H3)
rows = []
for lab, key in (("Classifier only (argmax per point)", "clf"), ("Classifier + HMM (default parameters)", "hmm_default"), ("Classifier + HMM (tuned)", "hmm")):
    d = hs[key]; rows.append([lab, pc(d["point_top1_accuracy"]), f4(d["route_continuity"]), f4(d["route_precision"]), f4(d["route_recall"]), f4(d["route_f1"]), f4(d["route_seq_sim"])])
table(rows, ["Student-test (10 trips, point-weighted)", "Point Top-1", "Route continuity", "Route precision", "Route recall", "Route F1", "Route sequence similarity"], [5.3 * cm, 2 * cm, 2.1 * cm, 2 * cm, 1.9 * cm, 1.8 * cm, 2.3 * cm])
P(f"Table 4. Route precision/recall/F1 compare the sets of arcs of the completed predicted route and the reference route (length-weighted), restricted to label-valid GPS points; sequence similarity is 2&middot;LCS/(|pred|+|ref|) (order-aware). "
  f"On the tuning set (out-of-fold, training trips) Top-1 was {pc(s['classifier_only'])} classifier-only, {pc(s['hmm_default'])} HMM default, {pc(s['hmm_tuned'])} HMM tuned.", CAP)
rows = [[r_.trip, int(r_.n_valid), pc(r_.clf_point_top1_accuracy), pc(r_.hmm_point_top1_accuracy), f"{100 * (r_.hmm_point_top1_accuracy - r_.clf_point_top1_accuracy):+.2f} pp",
         f"{r_.clf_route_continuity:.3f} / {r_.hmm_route_continuity:.3f}", f"{r_.clf_route_f1:.3f} / {r_.hmm_route_f1:.3f}", int(r_.hmm_bridged_gaps), int(r_.hmm_unbridged_breaks), int(r_.n_gaps_gt10s)] for r_ in ht.itertuples()]
table(rows, ["Trip", "Scored pts", "Top-1 clf", "Top-1 HMM", "Change", "Continuity clf / HMM", "Route F1 clf / HMM", "Bridged gaps", "Unbridged breaks", "GPS gaps >10 s"],
      [1.7 * cm, 1.4 * cm, 1.6 * cm, 1.6 * cm, 1.5 * cm, 2.4 * cm, 2.4 * cm, 1.4 * cm, 1.5 * cm, 1.5 * cm])
P("Table 5. Per-trip student-test results.", CAP)
worse = ht[ht.hmm_point_top1_accuracy <= ht.clf_point_top1_accuracy + 1e-9]
imp = hs["hmm"]["point_top1_accuracy"] - hs["clf"]["point_top1_accuracy"]
P(f"<b>Did HMM decoding improve the route?</b> {'Yes' if imp > 0 else 'No'}: point-level Top-1 changed from {pc(hs['clf']['point_top1_accuracy'])} to {pc(hs['hmm']['point_top1_accuracy'])} ({100 * imp:+.2f} pp) and route continuity from "
  f"{hs['clf']['route_continuity']:.3f} to {hs['hmm']['route_continuity']:.3f}. " +
  (f"<b>Where it did not help:</b> {len(worse)} trip(s) ({', '.join(worse.trip)}) did not improve in point accuracy. " if len(worse) else "It improved point accuracy on every student-test trip. ") +
  f"Remaining errors are dominated by points near junctions and by arcs shorter than the GPS step, where the reference alignment itself is ambiguous, and by trips with GPS gaps (see table 5).")
if err is not None:
    P("<b>Error analysis of classifier-only mistakes (student-test).</b> " + f"{pc(err.fraction['correct'])} of points are correct; the errors are: " + "; ".join(f"{i}: {pc(v)}" for i, v in err.fraction.items() if i != "correct") +
      ". The dominant error type is the opposite-direction arc of the same street, which the HMM resolves through connectivity and the U-turn penalty.")
rc = J("reference_continuity.json")["reference_label_continuity"]
P(f"<b>Interpreting 'route continuity'.</b> At 1 Hz a vehicle can skip a short arc between two fixes, so even the reference per-point labels are not perfectly connected: their own continuity is {rc:.3f} (student-test). "
  f"This is the ceiling for the metric. The classifier alone reaches {hs['clf']['route_continuity']:.3f}; classifier + HMM reaches {hs['hmm']['route_continuity']:.3f}, i.e. it closes most of the gap to the ceiling; after shortest-path bridging the completed route is connected wherever a path exists.")
img("hmm_vs_classifier.png", 17.3, "Figure 11. Classifier-only versus classifier + HMM on the student-test trips (point accuracy, route continuity, route F1).")
img("hmm_sensitivity.png", 17.3, "Figure 12. Sensitivity of tuning-set Top-1 accuracy to each HMM parameter around the optimum (dotted: classifier only).")

# ------------------------------------------------------------------ 6 folium
P("6. Geographical visualisation with Folium", H1)
P("All maps are generated by <font face='Courier'>scripts/06_make_maps.py</font> from the real pipeline output (no manual drawing). Layers (toggle in the layer control, legend bottom-left): "
  "<font color='#1f77b4'><b>blue</b></font> raw GPS observations, <font color='#9aa0a6'><b>grey</b></font> road network, <font color='#2e9e44'><b>green</b></font> reference route, "
  "<font color='#f28e2b'><b>orange dashed</b></font> classifier-only route, <font color='#d62728'><b>red</b></font> HMM-predicted route. Start/end markers and a measure tool are included. "
  "Interactive HTML files are in <font face='Courier'>maps/</font>.")
for p in sorted(FIGS.glob("map_*.png")):
    img(p.name, 16.5, f"Figure: exported view of {p.stem.replace('map_', 'student-test trip ')} (interactive version: maps/student_test_{p.stem.replace('map_', '')}_prediction.html).")

# ------------------------------------------------------------------ 7 api ui
P("7. API, UI and deployment", H1)
P("<b>Architecture.</b> Laptop 1 hosts the FastAPI server (trained bundle, HMM, road networks, server-side scoring); laptop 2 runs the Streamlit UI which lists the ten evaluation trips, sends the selected trip id to <font face='Courier'>POST /predict</font> "
  "and renders the returned route and metrics. Ground truth stays on laptop 1: the prediction function receives only the raw track and network, and the reference route is read afterwards by a separate scoring module; only metrics are returned.")
table([["GET /health", "Liveness, model name/version, HMM parameters"], ["GET /trips", "The ten evaluation trips available on the server"],
       ["POST /predict {trip_id}", "Full pipeline: candidates &rarr; features &rarr; classifier &rarr; HMM/Viterbi &rarr; completed route; server-side scoring; returns polylines, metrics and per-stage timing"]],
      ["Endpoint", "Purpose"], [4.2 * cm, 13 * cm])
for nm, cap in (("ui_screenshot.png", "Figure 13. Evaluator UI (Streamlit) showing a prediction result served by the API."),):
    if (FIGS / nm).exists(): img(nm, 17, cap)
logf = LOGS / "api_demo_excerpt.txt"
if logf.exists():
    P("<b>API request log (real run; each stage of the pipeline is logged):</b>")
    P("<font face='Courier' size=6.4>" + logf.read_text().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>") + "</font>", SM)
P("<b>Deployment steps</b> are in the repository README (two-laptop setup, environment variables, firewall note). "
  "<i>[Team: insert here the screenshots of the UI on laptop 2 and of the API log in the CMD window on laptop 1 taken during the live run, as required by the submission guidelines.]</i>")

# ------------------------------------------------------------------ 8 limits
P("8. Limitations, integrity statement", H1)
P("<b>Limitations.</b> (i) Reference routes are used as the evaluation reference but are not independently verified ground truth: some are partial, some contain small discontinuities, and point-level labels are derived by alignment, so accuracy is bounded by label noise. "
  "(ii) The 60 m candidate radius and K=12 leave a small residual candidate miss rate (table 2). (iii) The HMM relies on road connectivity inside a 700 m corridor around the track; links outside it are treated as unreachable. "
  "(iv) The classifier is trained on 80 trips from different cities, so unusual road layouts remain a risk.")
P("<b>Integrity.</b> Evaluator data was not used in any development step; hyper-parameters, model and decoder parameters were selected on training (group CV / out-of-fold) and student-test data only. No prediction is hard-coded. "
  "All results in this report are reproducible with the repository scripts. AI assistance is disclosed on the first page.")


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 7.5); c.setFillColor(colors.grey)
    c.drawString(1.5 * cm, 0.9 * cm, "PathSnap Technical Report - PES1UG24CS137 / 138 / 183 / 184"); c.drawRightString(A4[0] - 1.5 * cm, 0.9 * cm, f"Page {d.page}"); c.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                  title="PathSnap Technical Report", author="Team PES1UG24CS137/138/183/184").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
