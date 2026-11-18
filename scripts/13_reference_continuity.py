"""Ceiling for the continuity metric: at 1 Hz a vehicle can skip short arcs between two fixes, so even the reference
per-point labels are not perfectly 'connected'. Computed on the student-test trips."""
import _bootstrap  # noqa
import json, numpy as np
from psnap.config import *
from psnap.data import load_trip, list_trips
from psnap.network import RoadNetwork
from psnap.labeling import align_track_to_route, label_valid
from psnap.metrics import continuity
vals, w = [], []
for t in list_trips(DATA_ROOT / "student_test"):
    tr = load_trip(DATA_ROOT / "student_test", t, split="student_test"); net = RoadNetwork(tr)
    la, ld, _ = align_track_to_route(net, tr.route); v = label_valid(ld)
    vals.append(continuity(la[v], tr.arcs)); w.append(int(v.sum()))
out = dict(reference_label_continuity=float(np.average(vals, weights=w)), per_trip=dict(zip(list_trips(DATA_ROOT / "student_test"), map(float, vals))))
json.dump(out, open(RESULTS / "reference_continuity.json", "w"), indent=1); print(out["reference_label_continuity"])
