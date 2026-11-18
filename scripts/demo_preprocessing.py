"""TERMINAL DEMO 1 - data preprocessing, step by step on one real trip, then dataset-wide summary.
Usage:  python scripts/demo_preprocessing.py [--trip 00000004] [--split train]"""
import _bootstrap  # noqa
import argparse
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from psnap.config import *
from psnap.data import load_trip, list_trips
from psnap.network import RoadNetwork, seg_dist
from psnap.labeling import align_track_to_route
from psnap.candidates import generate_candidates
from psnap.features import build_features, FEATURES

ap = argparse.ArgumentParser(); ap.add_argument("--trip", default="00000004"); ap.add_argument("--split", default="train"); a = ap.parse_args()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
def step(n, t): print(f"\n{'=' * 80}\nSTEP {n}: {t}\n{'=' * 80}")

print(f"PathSnap - data preprocessing demo | raw data folder: {DATA_ROOT}")
print(f"Partitions: train = {len(list_trips(DATA_ROOT / 'train'))} trips, student_test = {len(list_trips(DATA_ROOT / 'student_test'))} trips (evaluator partition is never touched)")
tr = load_trip(DATA_ROOT / a.split, a.trip, split=a.split)

step(1, f"Load raw files of trip {a.trip} ({a.split})")
for name, arr, cols in (("track", tr.track, "lon, lat, time(s)"), ("nodes", tr.nodes, "lon, lat (row = node id)"), ("arcs", tr.arcs, "from_node, to_node (row = arc id)"), ("route", tr.route, "reference arc ids, in travel order")):
    print(f"\n.{name:<5} shape={arr.shape}  columns: {cols}")
    print(np.array2string(arr[:3], precision=6, suppress_small=True))

step(2, "Data-quality checks")
t = tr.track; dt = np.diff(t[:, 2])
print(f"GPS points            : {len(t)}")
print(f"NaN values            : {int(np.isnan(t).sum() + np.isnan(tr.nodes).sum())}")
print(f"timestamps increasing : {bool((dt > 0).all())}")
print(f"sampling interval     : {100 * np.mean(np.isclose(dt, 1)):.1f}% of steps are exactly 1 s | gaps > 10 s: {int((dt > 10).sum())} | largest gap: {dt.max():.0f} s")
print(f"lon range {t[:, 0].min():.4f}..{t[:, 0].max():.4f} | lat range {t[:, 1].min():.4f}..{t[:, 1].max():.4f}")
n = len(tr.nodes); comp = connected_components(csr_matrix((np.ones(len(tr.arcs)), (tr.arcs[:, 0], tr.arcs[:, 1])), shape=(n, n)), directed=False)[0]
print(f"road network          : {n} nodes, {len(tr.arcs)} directed arcs, {comp} connected component(s), isolated nodes: {int(((np.bincount(tr.arcs[:, 0], minlength=n) + np.bincount(tr.arcs[:, 1], minlength=n)) == 0).sum())}")
print(f"reference route       : {len(tr.route)} arcs, consecutive arcs connected: {100 * np.mean(tr.arcs[tr.route[:-1], 1] == tr.arcs[tr.route[1:], 0]):.1f}%")

step(3, "Project longitude/latitude (degrees) to metres")
net = RoadNetwork(tr)
print("Why: degrees are angular, Euclidean distance in degrees is wrong. We use a local azimuthal-equidistant projection per trip.")
print(pd.DataFrame({"lon": t[:3, 0], "lat": t[:3, 1], "x_m": net.px[:3], "y_m": net.py[:3]}).round(6).to_string(index=False))
d_m = np.hypot(net.px[1] - net.px[0], net.py[1] - net.py[0]); d_deg = np.hypot(*(t[1, :2] - t[0, :2]))
print(f"distance between GPS point 0 and 1: {d_m:.2f} metres  (naive 'degrees' distance would be {d_deg:.6f} - meaningless)")

step(4, "Create a label for every GPS point (.route has arcs, NOT which point is on which arc)")
la, ld, lp = align_track_to_route(net, tr.route)
valid = ld <= LABEL_MAX_DIST_M
print("Method: monotone dynamic-programming alignment of GPS points to the ordered route arcs (minimise sum of squared distances).")
idx = np.linspace(0, len(t) - 1, 8).astype(int)
print(pd.DataFrame({"gps_point": idx, "route_position": lp[idx], "reference_arc_id": la[idx], "dist_to_arc_m": ld[idx].round(1), "label_valid": valid[idx]}).to_string(index=False))
print(f"label-valid points (<= {LABEL_MAX_DIST_M:.0f} m from aligned arc): {valid.sum()} / {len(valid)} = {100 * valid.mean():.1f}%   | median GPS-to-road distance: {np.median(ld[valid]):.1f} m")

step(5, f"Candidate generation: {CAND_K} nearest arcs within {CAND_RADIUS_M:.0f} m of every GPS point")
pi, ai = generate_candidates(net)
P = np.c_[net.px, net.py]; dd, _, _ = seg_dist(P[pi], net.A[ai], net.B[ai])
for p in (0, len(t) // 2):
    m = pi == p
    print(f"\nGPS point {p}: true arc = {la[p]}")
    print(pd.DataFrame({"candidate_arc": ai[m], "dist_m": dd[m].round(1), "is_true_arc": np.where(ai[m] == la[p], "<-- TRUE", "")}).head(6).to_string(index=False))
hit = np.zeros(len(t), bool); hit[pi[ai == la[pi]]] = True
first = np.r_[True, pi[1:] != pi[:-1]]; nearest_ok = np.zeros(len(t), bool); nearest_ok[pi[first]] = (ai[first] == la[pi[first]])
print(f"\ncandidate rows: {len(pi)} ({len(pi) / len(t):.1f} per GPS point)")
print(f"true arc present among candidates (candidate hit rate): {100 * hit[valid].mean():.2f}%   -> miss rate {100 * (1 - hit[valid].mean()):.2f}%")
ab = RESULTS / "candidate_miss_ablation.csv"
glob = f" | over all 80 training trips: {100 * (1 - pd.read_csv(ab, index_col=0).loc[1, '60']):.1f}%" if ab.exists() else ""
print(f"'just take the nearest arc' is right only {100 * nearest_ok[valid].mean():.1f}% of the time on this trip{glob}")
print("   (two-way streets = 2 arcs with identical geometry, junctions, parallel roads)  -> we need a classifier to choose the right candidate")

step(6, f"Feature engineering: {len(FEATURES)} features per (GPS point, candidate arc) pair")
df = build_features(net, t[:, 2], pi, ai); df["y"] = (ai == la[pi]).astype(int)
print("Feature groups: geometric fit | position on arc | heading agreement | road topology | time gaps | temporal context | motion along arc")
print("\nAll features:", ", ".join(FEATURES))
show = ["pt", "arc", "d", "d_rank", "u", "cos_dh", "speed", "d_ctx_mean", "along_move", "y"]
print("\nExample rows (label y = 1 if the candidate is the true arc):")
print(df[df.pt == len(t) // 2][show].head(8).round(3).to_string(index=False))
print(f"\nrows for this trip: {len(df)} | positives: {int(df.y.sum())} ({100 * df.y.mean():.1f}%)")

step(7, "Whole dataset after preprocessing (all trips)")
for split in ("train", "student_test"):
    f = ART / "data" / f"{split}_pairs.parquet"
    if not f.exists():
        print(f"  {split}: not built yet -> run: python scripts/02_build_dataset.py"); continue
    pr = pd.read_parquet(f, columns=["trip", "y", "valid"]); pt = pd.read_parquet(ART / "data" / f"{split}_points.parquet")
    v = pt[pt.valid]
    print(f"  {split:<13}: trips={pr.trip.nunique():>3} | GPS points={len(pt):>7,} | label-valid points={100 * pt.valid.mean():.1f}% | (point,candidate) rows={len(pr):>9,} | "
          f"positive rate={100 * pr.y.mean():.1f}% | candidate miss={100 * (1 - v.in_cand.mean()):.2f}%")
print("\nTrain rows used for model fitting = label-valid rows of the 80 training trips. Splits are BY TRIP (no trip in two partitions).")
print("Next: python scripts/demo_train_models.py")
