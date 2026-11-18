# PathSnap — Distributed Map-Matching & Route Reconstruction

Reconstructs the road-network route of a vehicle from a noisy 1 Hz GPS trace.
Pipeline: **candidate generation → feature engineering → ML classifier (candidate probabilities) → HMM + Viterbi decoding → route**,
served by a **FastAPI** backend (Laptop 1) and a **Streamlit** evaluator UI (Laptop 2).

Team: PES1UG24CS137 Daksha S G · PES1UG24CS138 Darshan G S · PES1UG24CS183 Harsh Venkatsai Kalekar · PES1UG24CS184 Harsha K
Course: UE24CS352A Machine Learning, PES University (PathSnap hackathon)

> **AI-assistance disclosure.** This project was developed with assistance from an AI coding assistant (Claude Code).
> All experiments in `artifacts/` were executed for real by the scripts in this repository, and every number in the report is
> read from those result files. See `docs/AI_DISCLOSURE.md`.

---------------------------------------------------------------------------

## 1. Repository layout

```
src/psnap/            core library
  config.py           paths + constants (candidate radius/K, seeds)
  data.py             raw loading, local metric projection
  network.py          road network geometry, spatial index (STRtree), routing graph
  labeling.py         per-point labels from .route (monotone DP alignment)
  candidates.py       K nearest arcs within radius for every GPS point
  features.py         33 features per (GPS point, candidate arc) pair
  models.py           model zoo (10 models, 8 algorithm families)
  metrics.py          candidate-level, Top-1, route-level, continuity metrics
  hmm.py              HMM lattice + Viterbi decoder + route completion
  pipeline.py         end-to-end inference (raw files in, route out; never sees labels)
  evaluation.py       server-side scoring against the reference route
  viz.py              Folium map builder
scripts/              reproducible experiment pipeline (run in numeric order)
api/main.py           FastAPI backend (Laptop 1)
ui/app.py             Streamlit evaluator UI (Laptop 2)
artifacts/
  models/             saved model weights + HMM config (pathsnap_bundle.joblib is what the API loads)
  results/            CSV/JSON experiment records (inventory, 10-model comparison, Optuna trials, HMM results)
  figures/            all report figures (EDA, comparison, curves, confusion matrices, map screenshots)
  logs/               training + API logs
maps/                 interactive Folium HTML maps
reports/              technical report (PDF) + generator
docs/                 presentation guide, AI disclosure
```

## 2. Installation (macOS / Linux, Python 3.10+)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# macOS only: LightGBM/XGBoost need the OpenMP runtime
brew install libomp
```

The raw dataset is **not** stored in this repo. Point the code at it:

```bash
export PATHSNAP_DATA="/path/to/Student folder"      # contains train/ and student_test/
```

## 3. Reproduce the experiments (training laptop)

```bash
python scripts/01_eda.py              # dataset inventory, data-quality checks, EDA figures, candidate ablation, EDA Folium map
python scripts/02_build_dataset.py    # candidates + features + labels for train / student_test
python scripts/03_train_compare.py    # 10 models, 5-fold GROUP CV, student-test evaluation (resumable)
python scripts/04_bayes_opt.py auto 30   # Bayesian optimisation (Optuna TPE) of the best boosted model
python scripts/05_hmm_eval.py         # tune HMM/Viterbi, classifier-only vs +HMM, writes the deployable bundle
python scripts/07_model_plots.py      # comparison tables & figures
python scripts/08_training_curves.py  # real train/validation loss curves
python scripts/06_make_maps.py        # Folium maps + PNG exports
python scripts/11_error_analysis.py   # error types of the classifier-only predictions
python scripts/13_reference_continuity.py  # ceiling of the continuity metric
python scripts/09_make_report.py      # technical report PDF  (reports/PathSnap_Technical_Report.pdf)
python scripts/10_test_api.py http://127.0.0.1:8000   # end-to-end API test (needs the API running)
python scripts/12_ui_screenshot.py "<ui url>" out.png # optional: screenshot of the UI (macOS + Chrome)
python scripts/00_progress_summary.py # short interim summary PDF
```

## 3b. Main results (student-test partition, 10 trips)

| | Point Top-1 | Route F1 | Route continuity |
|---|---|---|---|
| Classifier only (tuned XGBoost) | 90.39 % | 0.869 | 0.780 |
| Classifier + HMM/Viterbi (tuned) | **95.84 %** | **0.992** | **0.911** (ceiling 0.932) |

Model comparison (10 models), Bayesian-optimisation record and per-trip results are in `artifacts/results/`
(`model_comparison.csv`, `bayes_trials.csv`, `hmm_vs_classifier_student_test.csv`).

Everything is seeded (`SEED=42`). The evaluator partition is never read by any of these scripts.

## 4. Deployment (two laptops, same Wi-Fi / hotspot)

**Laptop 1 — Model + API.** Needs the repo, `artifacts/models/pathsnap_bundle.joblib`, and the evaluation trip folders
(each `<id>/<id>.track|nodes|arcs|route`) in a directory set by `PATHSNAP_EVAL_DIR`
(defaults to the student-test split, used here as a stand-in during development).

```bash
export PATHSNAP_EVAL_DIR="/path/to/evaluator_trips"
uvicorn api.main:app --host 0.0.0.0 --port 8000        # keep this terminal visible: it prints the request log
```

Find Laptop 1's IP (`ipconfig getifaddr en0` on macOS) and allow incoming connections if the firewall asks.

**Laptop 2 — UI** (needs only `ui/requirements.txt`):

```bash
pip install -r ui/requirements.txt
export PATHSNAP_API="http://<laptop-1-ip>:8000"
streamlit run ui/app.py
```

**Quick API check from any machine:** `python scripts/10_test_api.py http://<laptop-1-ip>:8000`

API: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "<id>"}`. Interactive docs at `http://<laptop-1-ip>:8000/docs`.

### Integrity design
* The prediction pipeline (`psnap.pipeline.predict`) receives only the raw track + road network. Reference routes are read in
  `psnap.evaluation.evaluate`, **after** prediction finishes, and only metrics (never the reference) are returned to the UI.
* No hard-coded outputs: every `/predict` call runs candidates → features → model → HMM live and logs each stage.

## 5. Method summary

* **Projection:** per-trip azimuthal-equidistant projection (metres) – never Euclidean distance in degrees.
* **Labels:** `.route` gives the ordered arcs but not the arc of each GPS point; we align points to the route with a monotone
  dynamic-programming alignment. Points further than 50 m from their aligned arc are flagged label-invalid and excluded from supervision/scoring.
* **Candidates:** K=12 nearest arcs within 60 m (chosen from the candidate-miss ablation).
* **Classifier:** gradient-boosted trees, tuned with Optuna (TPE) using 5-fold **group** CV over the 80 training trips.
* **HMM:** states = candidate arcs; emission = classifier probability; transition = |route distance − GPS distance| on the road network;
  Viterbi decoding; gap/unreachable/U-turn handling; shortest-path route completion.
