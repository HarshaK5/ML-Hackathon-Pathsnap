"""Build the (GPS point, candidate arc) feature table for the train and student_test partitions.
NOTE: the evaluator partition is never touched by any training/development script."""
import _bootstrap  # noqa
import time, sys
import numpy as np, pandas as pd
from psnap.features import FEATURES
from joblib import Parallel, delayed
from psnap.config import *
from psnap.data import load_trip, list_trips
from psnap.network import RoadNetwork
from psnap.labeling import align_track_to_route
from psnap.candidates import generate_candidates
from psnap.features import build_features

OUT = ART / "data"; OUT.mkdir(exist_ok=True, parents=True)


def process(split, tid):
    tr = load_trip(DATA_ROOT / split, tid, split=split)
    net = RoadNetwork(tr)
    la, ld, lp = align_track_to_route(net, tr.route)
    pi, ai = generate_candidates(net)
    df = build_features(net, tr.track[:, 2], pi, ai)
    valid = ld <= LABEL_MAX_DIST_M
    df["y"] = (ai == la[pi]).astype(np.int8)
    df["valid"] = valid[pi]
    df.insert(0, "trip", tid)
    pts = pd.DataFrame({"trip": tid, "pt": np.arange(len(la)), "label_arc": la, "label_dist": ld,
                        "valid": valid, "t": tr.track[:, 2]})
    cnt = np.bincount(pi, minlength=len(la))
    pts["n_cand"] = cnt
    pts["in_cand"] = np.isin(np.arange(len(la)), pi[df["y"].values == 1])
    stats = dict(trip=tid, n_nodes=len(tr.nodes), n_arcs=len(tr.arcs), n_route=len(tr.route), points=len(la), pct_label_valid=100 * valid.mean(),
                 med_dist_m=float(np.median(ld[valid])) if valid.any() else float("nan"), cand_rows=len(df), cand_per_pt=len(df) / len(la),
                 positives=int(df["y"].sum()), cand_miss_pct=100 * (1 - pts[pts.valid].in_cand.mean()) if valid.any() else float("nan"))
    return df, pts, stats


if __name__ == "__main__":
    print("PathSnap preprocessing pipeline (this is the real pipeline that builds the model-training tables)")
    print(f"raw data: {DATA_ROOT}")
    print("per trip: (1) load .track/.nodes/.arcs/.route  (2) project lon/lat -> metres  (3) align GPS points to the reference route -> per-point label,")
    print(f"          mark label-invalid points (> {LABEL_MAX_DIST_M:.0f} m from aligned arc)  (4) candidate arcs: {CAND_K} nearest within {CAND_RADIUS_M:.0f} m  "
          f"(5) {len(FEATURES)} features per (point, candidate) pair  (6) y = 1 if candidate is the reference arc")
    for split in ["train", "student_test"]:
        t0 = time.time()
        ids = list_trips(DATA_ROOT / split)
        res = Parallel(n_jobs=-1)(delayed(process)(split, t) for t in ids)
        print(f"\n=== partition: {split} ({len(ids)} trips) ===")
        st = pd.DataFrame([r[2] for r in res]).round(2)
        st.columns = ["trip", "nodes", "arcs", "route_arcs", "gps_pts", "label_valid_%", "med_dist_m", "cand_rows", "cand/pt", "positives", "cand_miss_%"]
        pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
        print(st.to_string(index=False))
        pairs = pd.concat([r[0] for r in res], ignore_index=True)
        pts = pd.concat([r[1] for r in res], ignore_index=True)
        pairs.to_parquet(OUT / f"{split}_pairs.parquet", compression="zstd", compression_level=19)  # <100 MB so it fits on GitHub
        pts.to_parquet(OUT / f"{split}_points.parquet", compression="zstd", compression_level=19)
        v = pts[pts.valid]
        print(f"saved -> {OUT / (split + '_pairs.parquet')} , {OUT / (split + '_points.parquet')}")
        print(f"{split}: trips={len(ids)} points={len(pts)} pairs={len(pairs)} positives={int(pairs.y.sum())} "
              f"pos_rate={pairs.y.mean():.3f} valid_pts={pts.valid.mean():.4f} "
              f"cand_miss_valid={1 - v.in_cand.mean():.4f} time={time.time() - t0:.0f}s")
