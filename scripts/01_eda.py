"""Exploratory data analysis: inventory, data-quality checks, plots, candidate ablation, Folium map."""
import _bootstrap  # noqa
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from psnap.config import *
from psnap.data import load_trip, list_trips
from psnap.network import RoadNetwork, seg_dist
from psnap.labeling import align_track_to_route
from psnap.candidates import generate_candidates
from psnap.viz import build_folium_map

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
R_LIST, K_LIST = [20, 30, 40, 60, 80, 100], [1, 3, 5, 8, 12, 20]


def inspect(split, tid):
    tr = load_trip(DATA_ROOT / split, tid, split=split)
    t = tr.track
    net = RoadNetwork(tr)
    dt = np.diff(t[:, 2])
    nan = int(np.isnan(t).sum() + np.isnan(tr.nodes).sum())
    comps = connected_components(csr_matrix((np.ones(len(tr.arcs)), (tr.arcs[:, 0], tr.arcs[:, 1])),
                                            shape=(net.n_nodes, net.n_nodes)), directed=False)[0]
    iso = int((net.in_deg + net.out_deg == 0).sum())
    la, ld, _ = align_track_to_route(net, tr.route)
    v = ld <= LABEL_MAX_DIST_M
    pi, ai = generate_candidates(net, radius=100, k=40)
    d, _, _ = seg_dist(np.c_[net.px, net.py][pi], net.A[ai], net.B[ai])
    start = np.r_[0, np.flatnonzero(pi[1:] != pi[:-1]) + 1]
    cnt = np.diff(np.r_[start, len(pi)])
    rank = np.arange(len(pi)) - np.repeat(start, cnt)
    hit = ai == la[pi]
    dtrue = np.full(len(la), np.inf); rtrue = np.full(len(la), 10 ** 6)
    dtrue[pi[hit]] = d[hit]; rtrue[pi[hit]] = rank[hit]
    miss = {(R, K): float(1 - np.mean((dtrue[v] <= R) & (rtrue[v] < K))) for R in R_LIST for K in K_LIST}
    spd = np.hypot(np.diff(net.px), np.diff(net.py)) / np.maximum(dt, 1e-3)
    lon, lat = t[:, 0], t[:, 1]
    row = dict(split=split, trip=tid, n_points=len(t), duration_s=t[-1, 2] - t[0, 2],
               median_dt=np.median(dt), pct_dt_1s=100 * np.mean(np.isclose(dt, 1.0, atol=0.02)),
               n_gaps_gt10s=int((dt > 10).sum()), max_gap_s=dt.max(), nonmonotonic_t=int((dt <= 0).sum()),
               n_nan=nan, lon_min=lon.min(), lon_max=lon.max(), lat_min=lat.min(), lat_max=lat.max(),
               extent_km=float(np.hypot(net.px.max() - net.px.min(), net.py.max() - net.py.min()) / 1000),
               track_len_km=float(np.hypot(np.diff(net.px), np.diff(net.py)).sum() / 1000),
               n_nodes=net.n_nodes, n_arcs=net.n_arcs, n_components=int(comps), isolated_nodes=iso,
               pct_two_way=100 * float((net.reverse >= 0).mean()), route_arcs=len(tr.route),
               route_len_km=float(net.length[tr.route].sum() / 1000), route_connected=float(
                   np.mean(tr.arcs[tr.route[:-1], 1] == tr.arcs[tr.route[1:], 0])) if len(tr.route) > 1 else 1.0,
               route_repeated_arcs=len(tr.route) - len(set(tr.route.tolist())),
               label_valid_frac=float(v.mean()), label_dist_med=float(np.median(ld[v])) if v.any() else np.nan,
               label_dist_p99=float(np.percentile(ld[v], 99)) if v.any() else np.nan,
               speed_p50=float(np.median(spd)), speed_p99=float(np.percentile(spd, 99)))
    return row, miss, ld[v], spd[spd < 80], dt


if __name__ == "__main__":
    jobs = [(s, t) for s in ["train", "student_test"] for t in list_trips(DATA_ROOT / s)]
    res = Parallel(n_jobs=-1)(delayed(inspect)(s, t) for s, t in jobs)
    inv = pd.DataFrame([r[0] for r in res]); inv.to_csv(RESULTS / "dataset_inventory.csv", index=False)
    print(inv.groupby("split")[["n_points", "n_nodes", "n_arcs", "route_arcs", "n_gaps_gt10s"]].agg(["sum", "median", "max"]).T.to_string())
    print("NaN total", inv.n_nan.sum(), "| non-monotonic", inv.nonmonotonic_t.sum(), "| isolated nodes", inv.isolated_nodes.sum(),
          "| trips with >1 component", (inv.n_components > 1).sum(), "| route not connected", (inv.route_connected < 1).sum())
    # candidate ablation (train only)
    tr_idx = [i for i, (s, _) in enumerate(jobs) if s == "train"]
    nvalid = np.array([inv.n_points[i] * inv.label_valid_frac[i] for i in tr_idx])
    abl = pd.DataFrame({R: {K: float(np.average([res[i][1][(R, K)] for i in tr_idx], weights=nvalid)) for K in K_LIST} for R in R_LIST})
    abl.index.name = "K \\ radius_m"; abl.to_csv(RESULTS / "candidate_miss_ablation.csv"); print(abl.round(4).to_string())

    # ---------- figures ----------
    sp = inv.split.map({"train": "train (80)", "student_test": "student_test (10)"})
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for s, c in zip(["train", "student_test"], ["#4c78a8", "#e45756"]):
        q = inv[inv.split == s]
        ax[0, 0].hist(q.n_points, bins=25, alpha=.7, label=s, color=c)
        ax[0, 1].scatter(q.n_arcs, q.route_arcs, s=22, alpha=.8, label=s, color=c)
    ax[0, 0].set(title="GPS points per trip", xlabel="points", ylabel="trips"); ax[0, 0].legend()
    ax[0, 1].set(title="Road-network size vs reference-route length", xlabel="arcs in network (log)", ylabel="arcs in route", xscale="log"); ax[0, 1].legend()
    alld = np.concatenate([r[4] for r in res]); ax[0, 2].hist(np.clip(alld, 0, 12), bins=np.arange(0, 13) - .5, color="#59a14f")
    ax[0, 2].set(title=f"Sampling interval (clipped at 12s)\n{100 * np.mean(np.isclose(alld, 1, atol=.02)):.1f}% are exactly 1 s", xlabel="dt (s)", yscale="log")
    ld_all = np.concatenate([r[2] for r in res]); ax[1, 0].hist(np.clip(ld_all, 0, 50), bins=50, color="#f28e2b")
    ax[1, 0].set(title=f"GPS-to-reference-arc distance (label-valid pts)\nmedian {np.median(ld_all):.1f} m, p99 {np.percentile(ld_all, 99):.1f} m", xlabel="metres")
    spd = np.concatenate([r[3] for r in res]); ax[1, 1].hist(spd, bins=60, color="#b07aa1")
    ax[1, 1].set(title="GPS speed (m/s)", xlabel="m/s", yscale="log")
    ax[1, 2].bar(range(len(inv)), np.sort(inv.label_valid_frac.values), color="#76b7b2")
    ax[1, 2].set(title="Fraction of points with a valid reference label (sorted by trip)", xlabel="trip (sorted)", ylim=(0, 1.02))
    plt.tight_layout(); plt.savefig(FIGS / "eda_overview.png"); plt.close()

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for K in K_LIST:
        ax.plot(R_LIST, 100 * abl.loc[K].values, marker="o", label=f"K={K}")
    ax.axvline(CAND_RADIUS_M, ls=":", c="k"); ax.set(xlabel="search radius (m)", ylabel="candidate miss rate (%)", title="Candidate miss rate vs radius / K (train, label-valid points)", yscale="log"); ax.legend(ncol=3)
    plt.tight_layout(); plt.savefig(FIGS / "candidate_miss.png"); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    cen = inv.assign(lon=(inv.lon_min + inv.lon_max) / 2, lat=(inv.lat_min + inv.lat_max) / 2)
    for s, c in zip(["train", "student_test"], ["#4c78a8", "#e45756"]):
        q = cen[cen.split == s]; ax.scatter(q.lon, q.lat, s=18, color=c, label=s, alpha=.8)
    ax.set(title="Trip locations (centroid) - trips are worldwide, one road network per trip", xlabel="longitude", ylabel="latitude"); ax.legend()
    plt.tight_layout(); plt.savefig(FIGS / "trip_locations.png"); plt.close()

    # trajectory + network + reference route examples (matplotlib)
    ex = [("train", "00000005"), ("train", "00000002"), ("train", "00000054"), ("student_test", "00000052")]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.6))
    for (s, tid), a in zip(ex, axs):
        tr = load_trip(DATA_ROOT / s, tid, split=s); net = RoadNetwork(tr)
        pad = 300; sel = (net.A[:, 0] > net.px.min() - pad) & (net.A[:, 0] < net.px.max() + pad) & (net.A[:, 1] > net.py.min() - pad) & (net.A[:, 1] < net.py.max() + pad)
        for i in np.flatnonzero(sel)[:15000]:
            a.plot([net.A[i, 0], net.B[i, 0]], [net.A[i, 1], net.B[i, 1]], c="#c7c7c7", lw=.6, zorder=1)
        for i in tr.route:
            a.plot([net.A[i, 0], net.B[i, 0]], [net.A[i, 1], net.B[i, 1]], c="#2e9e44", lw=5, alpha=.9, zorder=2)
        a.scatter(net.px, net.py, s=1.5, c="#1f77b4", alpha=.6, zorder=3); a.set_aspect("equal"); a.set_title(f"{s} trip {tid}\n{len(tr.track)} pts, {len(tr.route)} route arcs"); a.axis("off")
    axs[0].scatter([], [], c="#1f77b4", s=8, label="GPS"); axs[0].plot([], [], c="#2e9e44", label="reference route"); axs[0].plot([], [], c="#c7c7c7", label="road network"); axs[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(FIGS / "eda_trajectories.png"); plt.close()

    # Folium map for EDA (train trip)
    tr = load_trip(DATA_ROOT / "train", "00000005", split="train")
    build_folium_map(tr, ref_route=tr.route, title="EDA - train trip 00000005 (GPS, network, reference route)").save(str(MAPS / "eda_train_00000005.html"))
    print("EDA done.")
