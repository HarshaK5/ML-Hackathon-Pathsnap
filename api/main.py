"""PathSnap FastAPI backend (runs on Laptop 1).

Endpoints
  GET  /health            liveness + model info
  GET  /trips             the evaluation trips available on this server (ids only)
  POST /predict           {"trip_id": "00000040"} -> runs the FULL pipeline server-side
                          (candidates -> features -> classifier -> HMM/Viterbi -> route), then scores it
                          against the server-side reference route and returns route + metrics.
Reference routes never leave the server and never enter the prediction pipeline."""
import sys, os, re, time, uuid, json, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from psnap.config import EVAL_DIR, MODELS, LOGS
from psnap.data import load_trip, list_trips
from psnap.hmm import HMMParams, complete_route
from psnap.pipeline import predict
from psnap.evaluation import evaluate
from psnap.viz import route_polyline

LOGS.mkdir(parents=True, exist_ok=True)
log = logging.getLogger("pathsnap.api")
log.setLevel(logging.INFO)
if not log.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS / "api.log")):
        h.setFormatter(fmt); log.addHandler(h)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BUNDLE = joblib.load(MODELS / "pathsnap_bundle.joblib")
MODEL = BUNDLE["model"]
PARAMS = HMMParams(**BUNDLE["hmm_params"])
TRIP_RE = re.compile(r"^\d{1,12}$")
log.info("Loaded model=%s version=%s | HMM params=%s | trips dir=%s", BUNDLE["kind"], BUNDLE["version"], PARAMS.as_dict(), EVAL_DIR)


def available_trips():
    return [t for t in list_trips(EVAL_DIR) if (Path(EVAL_DIR) / t / f"{t}.track").exists()]


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    t0 = time.time()
    client = request.client.host if request.client else "?"
    log.info("[%s] --> %s %s from %s", rid, request.method, request.url.path, client)
    response = await call_next(request)
    log.info("[%s] <-- %s %s status=%s in %.2fs", rid, request.method, request.url.path, response.status_code, time.time() - t0)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model": BUNDLE["kind"], "version": BUNDLE["version"], "hmm_params": PARAMS.as_dict(),
            "n_trips": len(available_trips())}


@app.get("/trips")
def trips():
    out = []
    for t in available_trips():
        n = sum(1 for _ in open(Path(EVAL_DIR) / t / f"{t}.track"))
        out.append({"trip_id": t, "n_gps_points": n})
    return {"trips": out}


def polylines(trip, arcs):
    return [[[float(a), float(b)] for a, b in ln] for ln in route_polyline(trip, arcs)]


@app.post("/predict")
def predict_trip(req: PredictRequest):
    tid = req.trip_id.strip()
    if not TRIP_RE.match(tid) or tid not in available_trips():
        log.warning("unknown trip_id=%r", tid)
        raise HTTPException(status_code=404, detail=f"trip {tid!r} not available")
    t0 = time.time()
    log.info("PREDICT trip=%s: loading raw GPS + road network (no labels)", tid)
    trip = load_trip(EVAL_DIR, tid, with_route=False)
    out = predict(trip.track, trip.nodes, trip.arcs, MODEL, PARAMS)
    T = out["timing"]
    log.info("PIPELINE trip=%s points=%d candidates=%d | features %.2fs classifier %.2fs lattice %.2fs viterbi %.2fs",
             tid, len(trip.track), out["n_cand"], T["features_s"], T["classifier_s"], T["lattice_s"], T["viterbi_s"])
    # ---- server-side scoring: the reference route is read only now, after prediction is finished ----
    ref = np.loadtxt(Path(EVAL_DIR) / tid / f"{tid}.route", dtype=np.int64, ndmin=1)
    ev = evaluate(out["net"], trip.arcs, ref, out)
    h, c = ev["hmm"], ev["classifier_only"]
    log.info("RESULT trip=%s | point Top-1: classifier-only=%.4f HMM=%.4f | route F1=%.4f continuity=%.4f | total %.2fs",
             tid, c["point_top1_accuracy"], h["point_top1_accuracy"], h["route_f1"], h["route_continuity"], time.time() - t0)
    step = max(1, len(trip.track) // 5000)
    clf_route, _, _ = complete_route(out["net"], out["clf_arc"])
    return {
        "trip_id": tid, "n_points": ev["n_points"], "n_label_valid_points": ev["n_label_valid_points"],
        "metrics": {"classifier_only": c, "hmm": h},
        "gps": [[float(la), float(lo)] for lo, la in trip.track[::step, :2]],
        "predicted_route": polylines(out["trip"], out["route"]),
        "classifier_only_route": polylines(out["trip"], clf_route),
        "predicted_route_arc_ids": [int(a) for a in out["route"]],
        "timing_seconds": {**{k: round(v, 3) for k, v in T.items()}, "request_total_s": round(time.time() - t0, 3)},
        "model": {"kind": BUNDLE["kind"], "version": BUNDLE["version"], "hmm_params": PARAMS.as_dict()},
    }
