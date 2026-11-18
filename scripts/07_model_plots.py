"""Comparison tables and figures from the recorded experiment results (artifacts/results/models/*.json)."""
import _bootstrap  # noqa
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from psnap.config import *

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
ORDER = ["LogReg", "GaussianNB", "KNN", "DecisionTree", "RandomForest", "ExtraTrees", "HistGB", "LightGBM", "XGBoost", "MLP"]
res = {k: json.load(open(RESULTS / "models" / f"{k}.json")) for k in ORDER if (RESULTS / "models" / f"{k}.json").exists()}
rows = []
for k, r in res.items():
    rows.append({"Model": k, "Algorithm family": r["family"], "Key configuration": ", ".join(f"{a}={b}" for a, b in r["config"].items()),
                 "CV Macro F1 (mean)": r["cv_mean"]["macro_f1"], "CV Macro F1 (std)": r["cv_std"]["macro_f1"],
                 "CV Top-1": r["cv_mean"]["top1"], "CV Route continuity": r["cv_mean"]["continuity"],
                 "Test Macro F1": r["test"]["macro_f1"], "Test Top-1": r["test"]["top1"], "Test Precision(+)": r["test"]["precision"],
                 "Test Recall(+)": r["test"]["recall"], "Test F1(+)": r["test"]["f1_pos"], "Test Log Loss": r["test"]["log_loss"],
                 "Test Route continuity": r["test"]["continuity"], "Train Macro F1": r["train"]["macro_f1"], "Train Top-1": r["train"]["top1"],
                 "CV Log Loss": r["cv_mean"]["log_loss"], "Fit time (s)": r["fit_seconds"], "Inference (ms/pt)": r["inference_ms_per_point"],
                 "Inference total (s)": r["inference_seconds_test"]})
T = pd.DataFrame(rows); T.to_csv(RESULTS / "model_comparison.csv", index=False)
print(T[["Model", "CV Macro F1 (mean)", "CV Top-1", "Test Macro F1", "Test Top-1", "Test Log Loss", "Test Route continuity", "Fit time (s)"]].round(4).to_string())
col = {m: c for m, c in zip(ORDER, plt.cm.tab10.colors)}
names = list(T.Model)
fig, ax = plt.subplots(figsize=(9, 3.8)); x = np.arange(len(T))
ax.bar(x, T["CV Macro F1 (mean)"], yerr=T["CV Macro F1 (std)"], capsize=3, color=[col[m] for m in names])
ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right"); ax.set(ylabel="Macro F1", title="5-fold group cross-validation Macro-F1 (mean ± std over folds)", ylim=(T["CV Macro F1 (mean)"].min() - .03, 1))
plt.tight_layout(); plt.savefig(FIGS / "cv_macro_f1.png"); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8)); w = .38
ax[0].bar(x - w / 2, T["Test Top-1"], w, label="Top-1 accuracy"); ax[0].bar(x + w / 2, T["Test Macro F1"], w, label="Macro F1")
ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30, ha="right"); ax[0].set(title="Student-test: Top-1 accuracy & Macro-F1", ylim=(.75, 1)); ax[0].legend(loc="lower right")
ax[1].bar(x, T["Test Log Loss"], color=[col[m] for m in names]); ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=30, ha="right"); ax[1].set(title="Student-test log-loss (lower is better)", yscale="log")
ax[2].bar(x - w / 2, T["Train Macro F1"], w, label="train (20k-point sample)"); ax[2].bar(x, T["CV Macro F1 (mean)"], w, label="CV"); ax[2].bar(x + w / 2, T["Test Macro F1"], w, label="student-test")
ax[2].set_xticks(x); ax[2].set_xticklabels(names, rotation=30, ha="right"); ax[2].set(title="Over/under-fitting check: Macro-F1 train vs CV vs test", ylim=(.75, 1.005)); ax[2].legend(loc="lower right", fontsize=7)
plt.tight_layout(); plt.savefig(FIGS / "test_comparison.png"); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].barh(names, T["Fit time (s)"], color=[col[m] for m in names]); ax[0].set(title="Final-fit time on 80 trips (s)", xscale="log")
ax[1].barh(names, T["Inference (ms/pt)"], color=[col[m] for m in names]); ax[1].set(title="Inference time (ms per GPS point)", xscale="log")
plt.tight_layout(); plt.savefig(FIGS / "efficiency.png"); plt.close()
n = len(res); fig, ax = plt.subplots(2, 5, figsize=(16, 6.5))
for a, (k, r) in zip(ax.ravel(), res.items()):
    cm = np.array(r["confusion_test"]); a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    a.set_title(k); a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["not true", "true arc"], fontsize=7); a.set_yticklabels(["not true", "true arc"], fontsize=7); a.grid(False)
for a in ax.ravel()[n:]: a.axis("off")
fig.suptitle("Student-test confusion matrices (candidate level; rows = actual, columns = predicted)"); plt.tight_layout(); plt.savefig(FIGS / "confusion_matrices.png"); plt.close()
print("figures saved")

# HMM sensitivity figure (re-drawn from the recorded CSV with readable ticks)
from matplotlib.ticker import ScalarFormatter
if (RESULTS / "hmm_sensitivity.csv").exists() and (RESULTS / "hmm_summary.json").exists():
    sens = pd.read_csv(RESULTS / "hmm_sensitivity.csv"); clf = json.load(open(RESULTS / "hmm_summary.json"))["tuning_set_top1"]["classifier_only"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2))
    for a, (k, g) in zip(ax, sens.groupby("param", sort=False)):
        a.plot(g.value, 100 * g.top1, marker="o"); a.axhline(100 * clf, ls=":", c="r", label="classifier only"); a.set(title=k, xlabel=k, ylabel="tuning-set top-1 (%)")
        if k in ("beta", "unreach_pen"):
            a.set_xscale("log"); a.set_xticks(list(g.value)); a.xaxis.set_major_formatter(ScalarFormatter()); a.minorticks_off(); a.tick_params(axis="x", labelsize=7)
    ax[0].legend(); plt.tight_layout(); plt.savefig(FIGS / "hmm_sensitivity.png"); plt.close()
