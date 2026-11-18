"""End-to-end inference pipeline used by the API. It takes ONLY raw GPS/network data - never labels."""
import time
import numpy as np
from .data import Trip
from .network import RoadNetwork
from .candidates import generate_candidates
from .features import build_features, FEATURES
from .metrics import top1_select, continuity
from .hmm import Lattice, viterbi, complete_route, HMMParams


def predict(trip_track, nodes, arcs, model, params: HMMParams):
    """Returns dict with per-point classifier-only / HMM arcs, completed route and timings."""
    T = {}
    t0 = time.time()
    trip = Trip("x", "", np.asarray(trip_track), np.asarray(nodes), np.asarray(arcs, dtype=np.int64), None)
    net = RoadNetwork(trip)
    pi, ai = generate_candidates(net)
    df = build_features(net, trip.track[:, 2], pi, ai)
    T["features_s"] = time.time() - t0
    t1 = time.time()
    p = model.predict_proba(df[FEATURES].values.astype(np.float32))[:, 1]
    T["classifier_s"] = time.time() - t1
    t2 = time.time()
    sel = top1_select(pi, p)
    clf_arc = ai[sel]
    lat = Lattice(net, trip.track[:, 2], pi, ai)
    T["lattice_s"] = time.time() - t2
    t3 = time.time()
    path = viterbi(lat, p, params)
    hmm_arc = ai[path]
    route, bridged, broken = complete_route(net, hmm_arc)
    T["viterbi_s"] = time.time() - t3
    T["total_s"] = time.time() - t0
    return dict(net=net, trip=trip, pi=pi, ai=ai, p=p, clf_arc=clf_arc, hmm_arc=hmm_arc, route=route,
                bridged=bridged, broken=broken, timing=T, n_cand=len(pi))
