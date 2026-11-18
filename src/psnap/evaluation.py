"""Server-side scoring. This is the ONLY place where reference routes are used, and only AFTER prediction
has finished: the prediction pipeline (pipeline.predict) never receives a route."""
import numpy as np
from .labeling import align_track_to_route, label_valid
from .metrics import continuity, route_metrics
from .hmm import complete_route


def evaluate(net, arcs, ref_route, out):
    """out: dict returned by pipeline.predict. Returns metrics for classifier-only and classifier+HMM."""
    la, ld, _ = align_track_to_route(net, np.asarray(ref_route))
    v = label_valid(ld)
    res = dict(n_points=int(len(la)), n_label_valid_points=int(v.sum()),
               # ceiling for the continuity metric: at 1 Hz a vehicle can skip short arcs between two fixes, so even the
               # reference labels are not perfectly 'connected' point to point
               reference_label_continuity=continuity(la[v], arcs) if v.any() else float("nan"))
    for name, arc in (("classifier_only", out["clf_arc"]), ("hmm", out["hmm_arc"])):
        acc = float((arc == la)[v].mean()) if v.any() else float("nan")
        route, bridged, broken = complete_route(net, arc[v]) if v.any() else ([], 0, 0)
        rm = route_metrics(route, ref_route, net.length)
        res[name] = dict(point_top1_accuracy=acc, route_continuity=continuity(arc[v], arcs),
                         route_arcs=len(route), bridged_gaps=bridged, unbridged_breaks=broken, **rm)
    return res
