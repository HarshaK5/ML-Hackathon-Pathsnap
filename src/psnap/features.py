"""Feature engineering: one row per (GPS point, candidate arc) pair."""
import numpy as np
import pandas as pd
from .network import RoadNetwork, seg_dist
from .config import CTX_OFFSETS

FEATURES = [
    "d", "lat_off", "d_rank", "d_min", "d_minus_min", "d_ratio", "n_cand",
    "u", "arc_len", "dist_to_start", "dist_to_end",
    "cos_dh", "sin_dh", "abs_dh", "speed", "heading_ok",
    "has_reverse", "deg_out_end", "deg_in_start", "deg_out_start",
    "gap_prev", "gap_next",
    "d_m3", "d_m2", "d_m1", "d_p1", "d_p2", "d_p3", "d_ctx_mean", "d_ctx_max",
    "along_move", "along_fwd_agree", "turn_rate",
]


def gps_motion(net: RoadNetwork, t):
    """Heading (rad), speed (m/s) from a +-2-point central difference; heading valid when moving."""
    N = len(net.px)
    i0 = np.clip(np.arange(N) - 2, 0, N - 1)
    i1 = np.clip(np.arange(N) + 2, 0, N - 1)
    dx, dy = net.px[i1] - net.px[i0], net.py[i1] - net.py[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-3)
    dist = np.hypot(dx, dy)
    speed = dist / dt
    ok = (dist > 6.0) & (speed > 1.0) & (dt < 30)
    return np.arctan2(dy, dx), speed, ok


def build_features(net: RoadNetwork, t: np.ndarray, pi: np.ndarray, ai: np.ndarray) -> pd.DataFrame:
    N = len(net.px)
    P = np.c_[net.px, net.py]
    A, B = net.A[ai], net.B[ai]
    d, u, lat = seg_dist(P[pi], A, B)

    # per-point candidate stats
    start = np.r_[0, np.flatnonzero(pi[1:] != pi[:-1]) + 1]
    counts = np.diff(np.r_[start, len(pi)])
    order_in_pt = np.lexsort((d, pi))                       # pi already sorted -> sort d within pt
    rank = np.empty(len(pi), dtype=np.int64)
    rk = np.arange(len(pi)) - np.repeat(start, counts)      # position by stored order
    # rank by distance within point
    o = np.lexsort((d, pi))
    rank[o] = np.arange(len(pi)) - np.repeat(start, counts)
    dmin = np.minimum.reduceat(d, start)
    d_min = np.repeat(dmin, counts)
    n_cand = np.repeat(counts, counts)

    heading, speed, ok = gps_motion(net, t)
    dh = heading[pi] - net.bearing[ai]
    cos_dh = np.where(ok[pi], np.cos(dh), 0.0)
    sin_dh = np.where(ok[pi], np.sin(dh), 0.0)
    abs_dh = np.where(ok[pi], np.abs(np.arctan2(np.sin(dh), np.cos(dh))), np.pi / 2)

    gap_prev = np.r_[0.0, np.diff(t)][pi]
    gap_next = np.r_[np.diff(t), 0.0][pi]

    f = {
        "d": d, "lat_off": lat, "d_rank": rank, "d_min": d_min, "d_minus_min": d - d_min,
        "d_ratio": d / (d_min + 1.0), "n_cand": n_cand,
        "u": u, "arc_len": net.length[ai], "dist_to_start": u * net.length[ai],
        "dist_to_end": (1 - u) * net.length[ai],
        "cos_dh": cos_dh, "sin_dh": sin_dh, "abs_dh": abs_dh, "speed": speed[pi], "heading_ok": ok[pi].astype(float),
        "has_reverse": (net.reverse[ai] >= 0).astype(float),
        "deg_out_end": net.out_deg[net.arcs[ai, 1]], "deg_in_start": net.in_deg[net.arcs[ai, 0]],
        "deg_out_start": net.out_deg[net.arcs[ai, 0]],
        "gap_prev": np.log1p(gap_prev), "gap_next": np.log1p(gap_next),
    }
    # temporal context: distance of neighbouring GPS points to the SAME candidate arc
    ctx = []
    uctx = {}
    for off in CTX_OFFSETS:
        j = np.clip(pi + off, 0, N - 1)
        dj, uj, _ = seg_dist(P[j], A, B)
        name = ("d_m%d" % -off) if off < 0 else ("d_p%d" % off)
        f[name] = dj
        ctx.append(dj)
        uctx[off] = uj
    ctx = np.stack(ctx, 1)
    f["d_ctx_mean"] = ctx.mean(1)
    f["d_ctx_max"] = ctx.max(1)
    # movement of the projected position along the arc over +-2 points (sign = travel direction)
    along = (uctx[2] - uctx[-2]) * net.length[ai]
    f["along_move"] = along
    f["along_fwd_agree"] = np.tanh(along / 5.0)
    # turn rate (heading change over +-4 points) -> junction / curve indicator
    j0, j1 = np.clip(pi - 4, 0, N - 1), np.clip(pi + 4, 0, N - 1)
    h = heading
    tr = np.arctan2(np.sin(h[j1] - h[j0]), np.cos(h[j1] - h[j0]))
    f["turn_rate"] = np.abs(tr)

    df = pd.DataFrame(f)[FEATURES].astype(np.float32)
    df.insert(0, "arc", ai)
    df.insert(0, "pt", pi)
    return df
