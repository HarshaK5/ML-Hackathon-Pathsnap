"""Real epoch/iteration-wise training histories (train vs validation log-loss) for the models that produce
them: LightGBM, XGBoost (boosting rounds) and the MLP (epochs). One CV fold (fold 0) is held out as validation.
Models without iterative training (KNN, trees, RF, ET, NB, LogReg-lbfgs) get no curve (by design)."""
import _bootstrap  # noqa
import json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from psnap.config import *
from psnap.models import load_split, xy, zoo

tr = load_split("train"); X, y = xy(tr)
a, b = next(iter(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, tr.trip.values)))
hist = {}
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
m.fit(X[a], y[a], eval_set=[(X[a], y[a]), (X[b], y[b])], eval_names=["train", "val"], eval_metric="binary_logloss")
hist["LightGBM"] = dict(train=m.evals_result_["train"]["binary_logloss"], val=m.evals_result_["val"]["binary_logloss"], unit="boosting round")
m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=SEED, n_jobs=-1, verbosity=0, eval_metric="logloss")
m.fit(X[a], y[a], eval_set=[(X[a], y[a]), (X[b], y[b])], verbose=False)
hist["XGBoost"] = dict(train=m.evals_result()["validation_0"]["logloss"], val=m.evals_result()["validation_1"]["logloss"], unit="boosting round")
sc = StandardScaler().fit(X[a]); Xa, Xb = sc.transform(X[a]), sc.transform(X[b])
net = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, random_state=SEED)
rs = np.random.RandomState(SEED); tr_l, va_l = [], []
for ep in range(40):
    perm = rs.permutation(len(Xa)); net.partial_fit(Xa[perm], y[a][perm], classes=[0, 1])
    sub = rs.choice(len(Xa), 200000, replace=False) if ep == 0 else sub
    tr_l.append(log_loss(y[a][sub], np.clip(net.predict_proba(Xa[sub])[:, 1], 1e-7, 1 - 1e-7), labels=[0, 1]))
    va_l.append(log_loss(y[b], np.clip(net.predict_proba(Xb)[:, 1], 1e-7, 1 - 1e-7), labels=[0, 1]))
hist["MLP"] = dict(train=tr_l, val=va_l, unit="epoch")
json.dump(hist, open(RESULTS / "training_histories.json", "w"))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for a_, (k, h) in zip(ax, hist.items()):
    a_.plot(h["train"], label="train", c="#4c78a8"); a_.plot(h["val"], label="validation (held-out trips)", c="#d62728")
    a_.set(title=f"{k}: log-loss per {h['unit']}", xlabel=h["unit"], ylabel="log-loss"); a_.legend()
plt.tight_layout(); plt.savefig(FIGS / "training_curves.png"); plt.close(); print("saved curves")
