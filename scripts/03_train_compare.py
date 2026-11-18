"""Train + compare 10 models: 5-fold GROUP cross-validation on the 80 training trips (no trip is
ever split across folds), final fit on all 80, evaluation on the 10 student-test trips.
Resumable: each model's result is stored in artifacts/results/models/<name>.json."""
import _bootstrap  # noqa
import json, time, sys
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix
from psnap.config import *
from psnap.data import load_trip
from psnap.models import zoo, load_split, xy
from psnap.metrics import candidate_metrics, top1_select, continuity

OUT = RESULTS / "models"; OUT.mkdir(parents=True, exist_ok=True)


def arcs_cache(df, split):
    return {t: np.loadtxt(DATA_ROOT / split / t / f"{t}.arcs", dtype=np.int64) for t in df.trip.unique()}


def point_level(df, p, arcs):
    """Observation-level Top-1 accuracy, plus classifier-only route continuity (arcs=None -> skipped)."""
    gid = df.gid.values
    sel = top1_select(gid, p)
    top1 = float(df.y.values[sel].mean())
    if arcs is None:
        return top1, float("nan"), len(sel)
    cont = []
    for t, g in df.iloc[sel].groupby("trip", sort=False):
        cont.append(continuity(g.arc.values, arcs[t]))
    return top1, float(np.mean(cont)), len(sel)


if __name__ == "__main__":
    only = sys.argv[1:]
    tr, te = load_split("train"), load_split("student_test")
    Xtr, ytr = xy(tr); Xte, yte = xy(te)
    A_tr, A_te = arcs_cache(tr, "train"), arcs_cache(te, "student_test")
    trips = tr.trip.values
    # training-set metrics (overfitting check) are measured on a FIXED random sample of 20k training GPS points
    rs = np.random.RandomState(SEED); keep_g = rs.choice(tr.gid.nunique(), 20000, replace=False)
    smask = np.isin(tr.gid.values, keep_g); tr_s = tr[smask]; Xs, ys = Xtr[smask], ytr[smask]
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xtr, ytr, trips))
    print(f"train rows {len(tr)} ({tr.gid.nunique()} pts) | test rows {len(te)} ({te.gid.nunique()} pts) | pos rate {ytr.mean():.3f}", flush=True)
    for name, (family, cfg, make) in zoo().items():
        if only and name not in only:
            continue
        f = OUT / f"{name}.json"
        if f.exists():
            print("skip (done)", name); continue
        t0 = time.time(); cv = []; oof = np.zeros(len(tr))
        for k, (a, b) in enumerate(folds):
            m = make().fit(Xtr[a], ytr[a]); p = m.predict_proba(Xtr[b])[:, 1]; oof[b] = p
            sub = tr.iloc[b]
            met = candidate_metrics(ytr[b], p)
            met["top1"], met["continuity"], _ = point_level(sub, p, A_tr)
            cv.append(met)
            print(f"  [{name}] fold {k}: macroF1={met['macro_f1']:.4f} top1={met['top1']:.4f} ll={met['log_loss']:.4f}", flush=True)
        # final fit on all 80 training trips
        t1 = time.time(); m = make().fit(Xtr, ytr); fit_s = time.time() - t1
        ptr = m.predict_proba(Xs)[:, 1]
        t2 = time.time(); pte = m.predict_proba(Xte)[:, 1]; inf_s = time.time() - t2
        train_met = candidate_metrics(ys, ptr); train_met["top1"], train_met["continuity"], _ = point_level(tr_s, ptr, None)
        test_met = candidate_metrics(yte, pte); test_met["top1"], test_met["continuity"], npts = point_level(te, pte, A_te)
        yh = (pte >= .5).astype(int)
        res = dict(model=name, family=family, config=cfg, cv=cv,
                   cv_mean={k: float(np.mean([c[k] for c in cv])) for k in cv[0]},
                   cv_std={k: float(np.std([c[k] for c in cv])) for k in cv[0]},
                   oof_overall=dict(**candidate_metrics(ytr, oof)),
                   train=train_met, test=test_met, fit_seconds=fit_s, cv_total_seconds=time.time() - t0,
                   inference_seconds_test=inf_s, inference_ms_per_point=1000 * inf_s / npts,
                   confusion_test=confusion_matrix(yte, yh).tolist())
        joblib.dump(m, MODELS / f"{name}.joblib")
        json.dump(res, open(f, "w"), indent=1, default=float)
        print(f"== {name}: CV macroF1 {res['cv_mean']['macro_f1']:.4f}±{res['cv_std']['macro_f1']:.4f} | CV top1 {res['cv_mean']['top1']:.4f} | "
              f"TEST macroF1 {test_met['macro_f1']:.4f} top1 {test_met['top1']:.4f} ll {test_met['log_loss']:.4f} | "
              f"train macroF1 {train_met['macro_f1']:.4f} | fit {fit_s:.0f}s | inf {inf_s:.2f}s | total {time.time() - t0:.0f}s", flush=True)
