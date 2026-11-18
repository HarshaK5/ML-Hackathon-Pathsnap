"""TERMINAL DEMO 2 - train the models live and print their accuracy.
  python scripts/demo_train_models.py                 # train all 10 models on the 80 training trips, test on the 10 student-test trips (~4 min)
  python scripts/demo_train_models.py --models LogReg XGBoost RandomForest   # only some models
  python scripts/demo_train_models.py --cv 5          # also add 5-fold GROUP cross-validation Macro-F1 (slower)
  python scripts/demo_train_models.py --saved         # instantly print the stored full results (5-fold CV + test) from our earlier runs"""
import _bootstrap  # noqa
import argparse, json, time
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from psnap.config import *
from psnap.models import zoo, load_split, xy
from psnap.metrics import candidate_metrics, top1_select

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*"); ap.add_argument("--cv", type=int, default=0); ap.add_argument("--saved", action="store_true")
a = ap.parse_args()
pd.set_option("display.width", 220)
LINE = "-" * 118


def explain():
    print("Metrics:  Macro-F1 = F1 averaged over both classes (true-road / not-true-road; only ~12% of candidates are true)")
    print("          Top-1    = per GPS point, is the highest-scored candidate road the reference road?  (the map-matching accuracy)")
    print("          LogLoss  = quality of the predicted probabilities (lower is better; they feed the HMM later)")


if a.saved:
    print("STORED RESULTS (5-fold GROUP cross-validation on 80 training trips, final fit on all 80, evaluated on 10 student-test trips)")
    explain(); print(LINE)
    c = pd.read_csv(RESULTS / "model_comparison.csv")
    t = c[["Model", "Algorithm family", "CV Macro F1 (mean)", "CV Macro F1 (std)", "CV Top-1", "Test Macro F1", "Test Top-1", "Test Log Loss", "Fit time (s)"]].copy()
    t = t.sort_values("Test Top-1", ascending=False)
    t["CV Top-1"] = (100 * t["CV Top-1"]).round(2).astype(str) + "%"; t["Test Top-1"] = (100 * t["Test Top-1"]).round(2).astype(str) + "%"
    print(t.round(4).to_string(index=False)); print(LINE)
    tv = json.load(open(RESULTS / "tuned_vs_untuned.json")); u, v = tv["untuned"], tv["tuned"]
    print(f"Best model: {c.loc[c['CV Macro F1 (mean)'].idxmax(), 'Model']} (highest cross-validated Macro-F1).  Bayesian tuning ({json.load(open(RESULTS / 'bayes_best.json'))['n_trials']} Optuna trials):")
    print(f"   CV Macro-F1 {u['cv_macro_f1']:.4f} -> {v['cv_macro_f1']:.4f} | test Top-1 {100 * u['test_top1']:.2f}% -> {100 * v['test_top1']:.2f}% | test log loss {u['test_log_loss']:.4f} -> {v['test_log_loss']:.4f}")
    raise SystemExit

t0 = time.time()
print("PathSnap - model training demo")
tr, te = load_split("train"), load_split("student_test")
Xtr, ytr = xy(tr); Xte, yte = xy(te)
print(f"Training data : {tr.trip.nunique()} trips | {tr.gid.nunique():,} GPS points | {len(tr):,} (point, candidate) rows | {Xtr.shape[1]} features | positives {100 * ytr.mean():.1f}%")
print(f"Test data     : {te.trip.nunique()} student-test trips | {te.gid.nunique():,} GPS points | {len(te):,} rows   (never used for fitting)")
explain(); print(LINE)
rs = np.random.RandomState(SEED); keep = rs.choice(tr.gid.nunique(), 20000, replace=False); sm = np.isin(tr.gid.values, keep)
folds = list(GroupKFold(n_splits=a.cv, shuffle=True, random_state=SEED).split(Xtr, ytr, tr.trip.values)) if a.cv else []
rows = []; names = [m for m in zoo() if not a.models or m in a.models]
for k, name in enumerate(names, 1):
    fam, cfg, make = zoo()[name]
    print(f"[{k}/{len(names)}] {name:<13} ({fam}) ... training", end="", flush=True)
    s = time.time(); m = make().fit(Xtr, ytr); fit = time.time() - s
    s = time.time(); p = m.predict_proba(Xte)[:, 1]; inf = time.time() - s
    met = candidate_metrics(yte, p); top1 = float(yte[top1_select(te.gid.values, p)].mean())
    ptr = m.predict_proba(Xtr[sm])[:, 1]; trf1 = candidate_metrics(ytr[sm], ptr)["macro_f1"]
    cvf1 = np.nan
    if folds:
        cvf1 = float(np.mean([candidate_metrics(ytr[b], make().fit(Xtr[a_], ytr[a_]).predict_proba(Xtr[b])[:, 1])["macro_f1"] for a_, b in folds]))
    print(f"  done in {fit:5.1f}s | TEST Top-1 {100 * top1:6.2f}% | Macro-F1 {met['macro_f1']:.4f} | log loss {met['log_loss']:.4f} | (train Macro-F1 {trf1:.4f})" + (f" | CV Macro-F1 {cvf1:.4f}" if folds else ""), flush=True)
    rows.append(dict(Model=name, Family=fam, **{"Train Macro-F1": trf1}, **({f"CV{a.cv} Macro-F1": cvf1} if folds else {}), **{"Test Macro-F1": met["macro_f1"], "Test Top-1": top1, "Test LogLoss": met["log_loss"], "Fit(s)": fit, "Predict(s)": inf}))
R = pd.DataFrame(rows).sort_values("Test Top-1", ascending=False).reset_index(drop=True)
print("\n" + LINE + "\nRESULTS on the 10 student-test trips (sorted by Top-1 accuracy)\n" + LINE)
out = R.copy(); out["Test Top-1"] = (100 * out["Test Top-1"]).round(2).astype(str) + "%"
print(out.round(4).to_string(index=False)); print(LINE)
b = R.iloc[0]
print(f"Best on student-test: {b.Model}  -> Top-1 accuracy {100 * b['Test Top-1']:.2f}%, Macro-F1 {b['Test Macro-F1']:.4f}, log loss {b['Test LogLoss']:.4f}")
print("Overfitting check: compare 'Train Macro-F1' (20k-point sample of training data) with 'Test Macro-F1' - a big gap means overfitting.")
print(f"Total time: {time.time() - t0:.0f}s.   Stored 5-fold CV results + Bayesian tuning: python scripts/demo_train_models.py --saved")
