"""Where does the classifier-only prediction go wrong? (student-test, final tuned model)"""
import _bootstrap  # noqa
import json
import numpy as np, pandas as pd, joblib
from psnap.config import *
from psnap.models import load_split, xy
from psnap.metrics import top1_select

kind = json.load(open(RESULTS / "bayes_best.json"))["model"]
te = load_split("student_test"); X, y = xy(te); m = joblib.load(MODELS / f"{kind}_tuned.joblib")
p = m.predict_proba(X)[:, 1]; sel = top1_select(te.gid.values, p)
pred = te.iloc[sel]
pts = pd.read_parquet(ART / "data" / "student_test_points.parquet"); lab = pts.set_index(["trip", "pt"]).label_arc
pred = pred.assign(label=lab.loc[list(zip(pred.trip, pred.pt))].values)
arcs = {t: np.loadtxt(DATA_ROOT / "student_test" / t / f"{t}.arcs", dtype=int) for t in te.trip.unique()}
cats = []
for t, a, l in zip(pred.trip, pred.arc, pred.label):
    if a == l: cats.append("correct"); continue
    fa, ta = arcs[t][a]; fl, tl = arcs[t][l]
    if fa == tl and ta == fl: cats.append("reverse arc (opposite direction, same street)")
    elif ta == fl or tl == fa: cats.append("adjacent arc (connected, same direction)")
    elif fa == fl or ta == tl: cats.append("sibling arc (shares a node)")
    else: cats.append("other (not directly connected)")
s = pd.Series(cats).value_counts(normalize=True)
s.to_csv(RESULTS / "error_analysis.csv", header=["fraction"]); print(s.round(4)); print("points:", len(pred))
