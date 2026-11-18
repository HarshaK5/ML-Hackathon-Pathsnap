"""Evaluation metrics: candidate-level, observation-level (Top-1), route-level, continuity."""
import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, accuracy_score, confusion_matrix


def top1_select(gid, p):
    """Row index of the highest-probability candidate for every GPS point (gid sorted/contiguous)."""
    order = np.lexsort((-p, gid))
    g = gid[order]
    first = np.r_[True, g[1:] != g[:-1]]
    return order[first]


def candidate_metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(macro_f1=f1_score(y, yh, average="macro"), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0), f1_pos=f1_score(y, yh, zero_division=0),
                accuracy=accuracy_score(y, yh), log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))


def continuity(seq, arcs):
    """Fraction of consecutive *distinct* arcs that are directly connected (end node == next start node)."""
    seq = np.asarray(seq)
    if len(seq) < 2:
        return 1.0
    keep = np.r_[True, seq[1:] != seq[:-1]]
    s = seq[keep]
    if len(s) < 2:
        return 1.0
    return float(np.mean(arcs[s[:-1], 1] == arcs[s[1:], 0]))


def lcs_len(a, b):
    """Longest common subsequence length of two integer sequences (O(len(a)*len(b)))."""
    if len(a) == 0 or len(b) == 0:
        return 0
    bi = {v: np.flatnonzero(np.asarray(b) == v) for v in set(a)}
    # classic DP with numpy row updates
    b = np.asarray(b)
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    for x in a:
        cur = prev.copy()
        m = (b == x)
        # cur[j+1] = prev[j]+1 where match ; then running max
        cand = np.where(m, prev[:-1] + 1, 0)
        cur[1:] = np.maximum(prev[1:], cand)
        cur = np.maximum.accumulate(cur)
        prev = cur
    return int(prev[-1])


def route_metrics(pred_route, ref_route, arc_len=None):
    """Route-level metrics between two arc-id sequences.
    precision/recall/F1 over the *sets* of arcs (length-weighted if arc_len given), plus
    sequence similarity = 2*LCS/(|pred|+|ref|) (order-aware)."""
    ps, rs = set(map(int, pred_route)), set(map(int, ref_route))
    inter = ps & rs
    w = (lambda s: float(sum(arc_len[list(s)]))) if arc_len is not None else (lambda s: float(len(s)))
    prec = w(inter) / max(w(ps), 1e-9)
    rec = w(inter) / max(w(rs), 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    lcs = lcs_len(list(map(int, pred_route)), list(map(int, ref_route)))
    sim = 2 * lcs / max(len(pred_route) + len(ref_route), 1)
    return dict(route_precision=prec, route_recall=rec, route_f1=f1, route_seq_sim=sim)
