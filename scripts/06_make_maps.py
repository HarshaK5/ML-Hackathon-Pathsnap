"""Interactive Folium maps (+ PNG exports) of the final system on student-test trips:
raw GPS, road network, reference route, classifier-only route, HMM-predicted route, with layer control + legend."""
import _bootstrap  # noqa
import subprocess, sys
from pathlib import Path
import numpy as np, joblib
from psnap.config import *
from psnap.data import load_trip, list_trips
from psnap.hmm import HMMParams, complete_route
from psnap.pipeline import predict
from psnap.evaluation import evaluate
from psnap.viz import build_folium_map

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def screenshot(html: Path, png: Path, w=1500, h=950):
    if not Path(CHROME).exists():
        return False
    try:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}", "--virtual-time-budget=25000",
                        f"--screenshot={png}", html.resolve().as_uri()], check=True, timeout=120, capture_output=True)
        return png.exists()
    except Exception as e:                      # noqa
        print("screenshot failed:", e); return False


if __name__ == "__main__":
    B = joblib.load(MODELS / "pathsnap_bundle.joblib"); hp = HMMParams(**B["hmm_params"])
    trips = sys.argv[1:] or ["00000052", "00000007", "00000056"]
    for tid in trips:
        tr = load_trip(DATA_ROOT / "student_test", tid, split="student_test")
        out = predict(tr.track, tr.nodes, tr.arcs, B["model"], hp)
        ev = evaluate(out["net"], tr.arcs, tr.route, out)
        clf_route, _, _ = complete_route(out["net"], out["clf_arc"])
        ttl = (f"student_test trip {tid} | point acc: classifier {100 * ev['classifier_only']['point_top1_accuracy']:.1f}% -> "
               f"HMM {100 * ev['hmm']['point_top1_accuracy']:.1f}%")
        m = build_folium_map(tr, pred_arcs=out["route"], ref_route=tr.route, classifier_arcs=clf_route, title=ttl)
        f = MAPS / f"student_test_{tid}_prediction.html"; m.save(str(f))
        ok = screenshot(f, FIGS / f"map_{tid}.png")
        print(tid, ttl, "| png" if ok else "| (no png)")
