"""Derive per-GPS-point reference arcs from the .route file.

The .route file lists arc ids in travel order but NOT which GPS point lies on which arc.
We recover point labels with a monotone alignment (dynamic programming): every point is
assigned a route position j(i) with j non-decreasing in i, minimising the sum of squared
point-to-arc distances. Routes are contiguous arc sequences, so the alignment is
well-posed. Points whose aligned arc is still far away (> LABEL_MAX_DIST_M) are flagged
as label-invalid (off-route/noisy) and excluded from supervised training/evaluation."""
import numpy as np
from .network import RoadNetwork, seg_dist
from .config import LABEL_MAX_DIST_M


def align_track_to_route(net: RoadNetwork, route: np.ndarray, cap_m: float = 150.0):
    """Returns label_arc (N,), label_dist (N,), label_pos (N,) route positions."""
    N, R = len(net.px), len(route)
    A, B = net.A[route], net.B[route]
    P = np.c_[net.px, net.py]
    cost = np.empty((N, R), dtype=np.float32)
    dmat = np.empty((N, R), dtype=np.float32)
    for j in range(R):                                   # distance of every point to route arc j
        d, _, _ = seg_dist(P, np.broadcast_to(A[j], P.shape), np.broadcast_to(B[j], P.shape))
        dmat[:, j] = d
    cost = np.minimum(dmat, cap_m) ** 2
    f = cost[0].copy()
    back = np.zeros((N, R), dtype=np.int32)
    idx = np.arange(R, dtype=np.int32)
    for i in range(1, N):
        acc = np.minimum.accumulate(f)
        arg = np.maximum.accumulate(np.where(f == acc, idx, 0))
        back[i] = arg
        f = cost[i] + acc
    j = np.empty(N, dtype=np.int64)
    j[-1] = int(np.argmin(f))
    for i in range(N - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    label_arc = route[j]
    label_dist = dmat[np.arange(N), j].astype(np.float64)
    return label_arc, label_dist, j


def label_valid(label_dist):
    return label_dist <= LABEL_MAX_DIST_M
