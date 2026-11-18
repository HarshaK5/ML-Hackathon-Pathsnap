"""Bayesian hyper-parameter optimisation (Optuna TPE) of the best boosted model.
Objective  : mean 5-fold GROUP-CV macro-F1 on the 80 training trips (same folds as the model comparison).
Evaluation : tuned vs untuned on the 10 student-test trips. The evaluator partition is never used."""
import _bootstrap  # noqa
import json, time, sys
import numpy as np, pandas as pd, joblib, optuna
import lightgbm as lgb, xgboost as xgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from psnap.config import *
from psnap.models import load_split, xy, zoo
from psnap.metrics import candidate_metrics, top1_select
from psnap.features import FEATURES

N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def make(kind, p):
    if kind == "LightGBM":
        return lgb.LGBMClassifier(**p, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1)
    return xgb.XGBClassifier(**p, tree_method="hist", random_state=SEED, n_jobs=-1, verbosity=0)


def space(kind, tr):
    if kind == "LightGBM":
        return dict(n_estimators=tr.suggest_int("n_estimators", 100, 500, step=50),
                    learning_rate=tr.suggest_float("learning_rate", 0.03, 0.3, log=True),
                    num_leaves=tr.suggest_int("num_leaves", 15, 255, log=True),
                    min_child_samples=tr.suggest_int("min_child_samples", 10, 200, log=True),
                    subsample=tr.suggest_float("subsample", 0.5, 1.0),
                    colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0),
                    reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    reg_alpha=tr.suggest_float("reg_alpha", 1e-3, 10.0, log=True))
    return dict(n_estimators=tr.suggest_int("n_estimators", 100, 500, step=50),
                learning_rate=tr.suggest_float("learning_rate", 0.03, 0.3, log=True),
                max_depth=tr.suggest_int("max_depth", 3, 12), min_child_weight=tr.suggest_float("min_child_weight", 1, 50, log=True),
                subsample=tr.suggest_float("subsample", 0.5, 1.0), colsample_bytree=tr.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 10.0, log=True), reg_alpha=tr.suggest_float("reg_alpha", 1e-3, 10.0, log=True))


if __name__ == "__main__":
    # model to tune: best CV macro-F1 among the boosted models of the comparison (or forced via argv[1])
    res = {f.stem: json.load(open(f)) for f in (RESULTS / "models").glob("*.json")}
    boosted = {k: v for k, v in res.items() if k in ("LightGBM", "XGBoost")}
    kind = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "auto" else max(boosted, key=lambda k: boosted[k]["cv_mean"]["macro_f1"])
    print("tuning", kind, "trials", N_TRIALS, flush=True)
    tr, te = load_split("train"), load_split("student_test")
    Xtr, ytr = xy(tr); Xte, yte = xy(te)
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xtr, ytr, tr.trip.values))

    def objective(trial):
        p = space(kind, trial); f1s, t1s = [], []
        for k, (a, b) in enumerate(folds):
            m = make(kind, p).fit(Xtr[a], ytr[a]); pr = m.predict_proba(Xtr[b])[:, 1]
            met = candidate_metrics(ytr[b], pr); sel = top1_select(tr.gid.values[b], pr)
            f1s.append(met["macro_f1"]); t1s.append(float(ytr[b][sel].mean()))
            trial.report(float(np.mean(f1s)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_top1", float(np.mean(t1s))); trial.set_user_attr("cv_macro_f1_std", float(np.std(f1s)))
        return float(np.mean(f1s))

    db = RESULTS / "optuna_study.db"
    if db.exists():
        db.unlink()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1),
                                storage=f"sqlite:///{db}", study_name=f"{kind}_macroF1")
    base = zoo()[kind][1]            # warm start: first trial = the untuned configuration
    study.enqueue_trial({k: base[k] for k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree")
                         if kind == "LightGBM" and k in base} if kind == "LightGBM" else
                        {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.8})
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False,
                   callbacks=[lambda s, t: print(f"trial {t.number:3d} {t.state.name:9s} value={t.value if t.value is not None else float('nan'):.5f} best={s.best_value:.5f}", flush=True)])
    opt_s = time.time() - t0
    df = study.trials_dataframe(); df.to_csv(RESULTS / "bayes_trials.csv", index=False)
    best = study.best_params
    json.dump(dict(model=kind, best_params=best, best_cv_macro_f1=study.best_value, n_trials=len(study.trials),
                   n_complete=sum(t.state.name == "COMPLETE" for t in study.trials), n_pruned=sum(t.state.name == "PRUNED" for t in study.trials),
                   optimisation_seconds=opt_s, sampler="TPE (Optuna)", pruner="MedianPruner", seed=SEED), open(RESULTS / "bayes_best.json", "w"), indent=1)

    # refit tuned model on all 80 training trips; OOF predictions (for HMM tuning); test evaluation
    t1 = time.time(); m = make(kind, best).fit(Xtr, ytr); fit_s = time.time() - t1
    t2 = time.time(); pte = m.predict_proba(Xte)[:, 1]; inf_s = time.time() - t2
    sel = top1_select(te.gid.values, pte); npts = len(sel)
    out = candidate_metrics(yte, pte); out["top1"] = float(yte[sel].mean())
    oof = np.zeros(len(tr))
    for a, b in folds:
        oof[b] = make(kind, best).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
    oofm = candidate_metrics(ytr, oof); oofm["top1"] = float(ytr[top1_select(tr.gid.values, oof)].mean())
    pd.DataFrame({"trip": tr.trip.values, "pt": tr.pt.values, "arc": tr.arc.values, "p": oof}).to_parquet(ART / "data" / "oof_tuned.parquet")
    joblib.dump(m, MODELS / f"{kind}_tuned.joblib")
    base_res = res[kind]
    cmp_ = dict(model=kind,
                untuned=dict(cv_macro_f1=base_res["cv_mean"]["macro_f1"], cv_top1=base_res["cv_mean"]["top1"], test_top1=base_res["test"]["top1"],
                             test_macro_f1=base_res["test"]["macro_f1"], test_log_loss=base_res["test"]["log_loss"],
                             inference_ms_per_point=base_res["inference_ms_per_point"], fit_seconds=base_res["fit_seconds"]),
                tuned=dict(cv_macro_f1=study.best_value, cv_top1=study.best_trial.user_attrs["cv_top1"], test_top1=out["top1"],
                           test_macro_f1=out["macro_f1"], test_log_loss=out["log_loss"], inference_ms_per_point=1000 * inf_s / npts, fit_seconds=fit_s),
                oof_tuned=oofm, tuned_test_full=out)
    json.dump(cmp_, open(RESULTS / "tuned_vs_untuned.json", "w"), indent=1, default=float)
    print(json.dumps(cmp_, indent=1, default=float))

    # plots: optimisation history + parameter importance + parallel of best
    comp = df[df.state == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].scatter(df.number, df.value, c=np.where(df.state == "COMPLETE", "#4c78a8", "#bbbbbb"), s=22)
    ax[0].plot(comp.number, comp.value.cummax(), c="#d62728", label="best so far"); ax[0].legend()
    ax[0].set(title=f"Bayesian optimisation history ({kind}, TPE)", xlabel="trial", ylabel="CV macro-F1 (5-fold group CV)")
    imp = optuna.importance.get_param_importances(study)
    ax[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1], color="#59a14f"); ax[1].set(title="Hyper-parameter importance (fANOVA)")
    plt.tight_layout(); plt.savefig(FIGS / "bayes_opt.png"); plt.close()
    print("done")
