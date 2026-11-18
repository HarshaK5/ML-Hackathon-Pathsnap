"""Tune the HMM/Viterbi decoder parameters and compare classifier-only vs classifier+HMM.
Tuning data : OUT-OF-FOLD classifier probabilities of 30 fully-labelled TRAINING trips (no leakage).
Reporting   : the 10 student-test trips, full pipeline from raw files. Evaluator partition never used."""
import _bootstrap  # noqa
import json, time
import numpy as np, pandas as pd, joblib, optuna
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from psnap.config import *
from psnap.data import load_trip, list_trips
from psnap.network import RoadNetwork
from psnap.labeling import align_track_to_route
from psnap.candidates import generate_candidates
from psnap.hmm import Lattice, viterbi, HMMParams
from psnap.metrics import top1_select
from psnap.pipeline import predict
from psnap.evaluation import evaluate
from psnap.features import FEATURES

optuna.logging.set_verbosity(optuna.logging.WARNING)
N_TRIALS, N_TUNE_TRIPS = 60, 30

if __name__ == "__main__":
    best = json.load(open(RESULTS / "bayes_best.json")); kind = best["model"]
    model = joblib.load(MODELS / f"{kind}_tuned.joblib")
    oof = pd.read_parquet(ART / "data" / "oof_tuned.parquet")
    inv = pd.read_csv(RESULTS / "dataset_inventory.csv", dtype={"trip": str})
    inv["trip"] = inv.trip.str.zfill(8)
    cand = inv[(inv.split == "train") & (inv.label_valid_frac == 1.0) & (inv.n_points <= 6000)]
    tune_trips = sorted(cand.sample(N_TUNE_TRIPS, random_state=SEED).trip.tolist())
    print("tuning trips:", len(tune_trips), flush=True)

    data = []
    t0 = time.time()
    for tid in tune_trips:
        tr = load_trip(DATA_ROOT / "train", tid, split="train"); net = RoadNetwork(tr)
        la, ld, _ = align_track_to_route(net, tr.route)
        pi, ai = generate_candidates(net)
        o = oof[oof.trip == tid]
        assert len(o) == len(pi) and (o.arc.values == ai).all(), f"row mismatch {tid}"
        lat = Lattice(net, tr.track[:, 2], pi, ai)
        lat.g = None; lat._cache = {}; net._tree = None
        data.append((tid, lat, o.p.values, la, pi, ai, tr.arcs))
    print(f"lattices built in {time.time() - t0:.0f}s", flush=True)
    npts = sum(len(d[3]) for d in data)

    def score(params, which="hmm"):
        correct = 0
        for tid, lat, p, la, pi, ai, arcs in data:
            if which == "clf":
                arc = ai[top1_select(pi, p)]
            else:
                arc = ai[viterbi(lat, p, params)]
            correct += int((arc == la).sum())
        return correct / npts

    clf_acc = score(None, "clf"); default_acc = score(HMMParams())
    print(f"tuning-set (OOF) top-1: classifier-only {clf_acc:.4f} | HMM default params {default_acc:.4f}", flush=True)

    def objective(tr):
        p = HMMParams(beta=tr.suggest_float("beta", 2, 80, log=True), lam=tr.suggest_float("lam", 0.2, 3.0, log=True),
                      unreach_pen=tr.suggest_float("unreach_pen", 1, 40, log=True), uturn_pen=tr.suggest_float("uturn_pen", 0, 20),
                      switch_pen=tr.suggest_float("switch_pen", 0, 3), gap_thresh=tr.suggest_float("gap_thresh", 8, 60))
        return score(p)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(HMMParams().as_dict())
    study.optimize(objective, n_trials=N_TRIALS)
    study.trials_dataframe().to_csv(RESULTS / "hmm_tuning_trials.csv", index=False)
    hp = HMMParams(**study.best_params)
    print("best HMM params", hp, "tuning-set top-1", study.best_value, flush=True)

    # sensitivity (one parameter at a time around the optimum)
    sweeps = {"beta": [2, 4, 8, 12, 20, 40, 80], "lam": [0.2, 0.5, 1, 2, 3], "unreach_pen": [1, 3, 6, 12, 25, 40], "uturn_pen": [0, 2, 5, 10, 20]}
    sens = []
    for k, vals in sweeps.items():
        for v in vals:
            d = hp.as_dict(); d[k] = v; sens.append(dict(param=k, value=v, top1=score(HMMParams(**d))))
    pd.DataFrame(sens).to_csv(RESULTS / "hmm_sensitivity.csv", index=False)
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    for a, (k, g) in zip(ax, pd.DataFrame(sens).groupby("param", sort=False)):
        a.plot(g.value, 100 * g.top1, marker="o"); a.axhline(100 * clf_acc, ls=":", c="r", label="classifier only"); a.set(title=k, xlabel=k, ylabel="tuning-set top-1 (%)")
        if k in ("beta", "unreach_pen"): a.set_xscale("log")
    ax[0].legend(); plt.tight_layout(); plt.savefig(FIGS / "hmm_sensitivity.png"); plt.close()

    # ---------- student-test: full pipeline from raw files ----------
    rows = []; tot = dict(n=0, clf=0, hmm=0, hmm_def=0)
    for tid in list_trips(DATA_ROOT / "student_test"):
        tr = load_trip(DATA_ROOT / "student_test", tid, split="student_test")
        r_def = predict(tr.track, tr.nodes, tr.arcs, model, HMMParams())
        r = predict(tr.track, tr.nodes, tr.arcs, model, hp)
        ev = evaluate(r["net"], tr.arcs, tr.route, r); ev_def = evaluate(r_def["net"], tr.arcs, tr.route, r_def)
        row = dict(trip=tid, n_points=ev["n_points"], n_valid=ev["n_label_valid_points"], time_s=r["timing"]["total_s"],
                   n_gaps_gt10s=int((np.diff(tr.track[:, 2]) > 10).sum()))
        for pre, e in (("clf", ev["classifier_only"]), ("hmm", ev["hmm"]), ("hmm_default", ev_def["hmm"])):
            for k, v in e.items():
                row[f"{pre}_{k}"] = v
        rows.append(row)
        print(f"{tid}: top1 clf {row['clf_point_top1_accuracy']:.4f} -> hmm {row['hmm_point_top1_accuracy']:.4f} | "
              f"cont {row['clf_route_continuity']:.3f} -> {row['hmm_route_continuity']:.3f} | route F1 {row['clf_route_f1']:.3f} -> {row['hmm_route_f1']:.3f}", flush=True)
    T = pd.DataFrame(rows); T.to_csv(RESULTS / "hmm_vs_classifier_student_test.csv", index=False)
    w = T.n_valid
    summ = {"tuning_set_top1": dict(classifier_only=clf_acc, hmm_default=default_acc, hmm_tuned=study.best_value, n_trips=N_TUNE_TRIPS, n_points=npts),
            "hmm_params": hp.as_dict(), "hmm_tuning": dict(n_trials=N_TRIALS, sampler="TPE", objective="pooled point top-1 accuracy on OOF probabilities")}
    for pre in ("clf", "hmm_default", "hmm"):
        summ[pre] = {k: float(np.average(T[f"{pre}_{k}"], weights=w)) for k in
                     ("point_top1_accuracy", "route_continuity", "route_precision", "route_recall", "route_f1", "route_seq_sim")}
        summ[pre]["macro_trip_top1"] = float(T[f"{pre}_point_top1_accuracy"].mean())
    json.dump(summ, open(RESULTS / "hmm_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8)); x = np.arange(len(T)); wd = .38
    for a, (c, ttl) in zip(ax, [("point_top1_accuracy", "Point-level Top-1 accuracy"), ("route_continuity", "Route continuity"), ("route_f1", "Route F1 (arc set, length-weighted)")]):
        a.bar(x - wd / 2, T[f"clf_{c}"], wd, label="classifier only", color="#f28e2b"); a.bar(x + wd / 2, T[f"hmm_{c}"], wd, label="classifier + HMM", color="#d62728")
        a.set_xticks(x); a.set_xticklabels(T.trip.str[-3:], rotation=0); a.set(title=ttl, xlabel="student-test trip", ylim=(max(0, T[[f'clf_{c}', f'hmm_{c}']].min().min() - .05), 1.005))
    ax[0].legend(loc="lower left"); plt.tight_layout(); plt.savefig(FIGS / "hmm_vs_classifier.png"); plt.close()

    # export the deployable bundle (model + decoder config) for the API
    joblib.dump(dict(model=model, kind=kind, features=FEATURES, hmm_params=hp.as_dict(),
                     candidate=dict(radius=CAND_RADIUS_M, k=CAND_K), version="1.0"), MODELS / "pathsnap_bundle.joblib")
    json.dump(hp.as_dict(), open(MODELS / "hmm_params.json", "w"), indent=1)
    print("bundle saved")
