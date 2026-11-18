"""Candidate generation: for every GPS point, the K nearest road arcs within a radius."""
import numpy as np
import shapely
from .network import RoadNetwork, seg_dist
from .config import CAND_RADIUS_M, CAND_K, CAND_FALLBACK_K


def generate_candidates(net: RoadNetwork, radius=CAND_RADIUS_M, k=CAND_K, fallback_k=CAND_FALLBACK_K):
    """Returns (pt_idx, arc_idx) integer arrays (sorted by point, then distance)."""
    N = len(net.px)
    pts = shapely.points(net.px, net.py)
    pi, ai = net.tree.query(pts, predicate="dwithin", distance=radius)
    P = np.c_[net.px, net.py]
    d, _, _ = seg_dist(P[pi], net.A[ai], net.B[ai])
    order = np.lexsort((d, pi))
    pi, ai, d = pi[order], ai[order], d[order]
    # rank within point -> keep top-k
    start = np.r_[0, np.flatnonzero(pi[1:] != pi[:-1]) + 1]
    counts = np.diff(np.r_[start, len(pi)])
    rank = np.arange(len(pi)) - np.repeat(start, counts)
    keep = rank < k
    pi, ai = pi[keep], ai[keep]
    # fallback: points without any arc in radius -> nearest few arcs at any distance
    have = np.zeros(N, bool)
    have[pi] = True
    miss = np.flatnonzero(~have)
    if len(miss):
        qi, ti = net.tree.query_nearest(pts[miss], all_matches=False, return_distance=False)
        extra_p, extra_a = miss[qi], ti
        # widen: also add reverse arcs / next-nearest by querying a grown buffer
        more_p, more_a = [], []
        for m in miss:
            r = 200.0
            while True:
                cand = net.tree.query(pts[m], predicate="dwithin", distance=r)
                if len(cand) >= 1 or r > 5000:
                    break
                r *= 2
            dd, _, _ = seg_dist(np.repeat(P[m][None], len(cand), 0), net.A[cand], net.B[cand])
            sel = cand[np.argsort(dd)[:fallback_k]]
            more_p += [m] * len(sel); more_a += list(sel)
        pi = np.r_[pi, np.array(more_p, dtype=pi.dtype)]
        ai = np.r_[ai, np.array(more_a, dtype=ai.dtype)]
        o = np.lexsort((np.arange(len(pi)), pi))
        pi, ai = pi[o], ai[o]
    return pi.astype(np.int64), ai.astype(np.int64)
