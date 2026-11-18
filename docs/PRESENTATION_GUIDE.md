# PathSnap — How to present it

Numbers below are the final measured results (student-test = the 10 development trips). If you re-run anything, re-check them against
`artifacts/results/` before you speak.

## 0. The 30-second story (memorise this)

> "GPS points drift off the road. Given the noisy track and the road network, we reconstruct the roads the car really drove.
> We generate candidate roads for every GPS point, score each (point, road) pair with a machine-learning classifier,
> and then an HMM with Viterbi decoding uses road connectivity to pick the most coherent route.
> We compared 10 models, tuned the best one with Bayesian optimisation, and the HMM raised point accuracy from 90.4 % to 95.8 % and route F1 from 0.87 to 0.99."

## 1. Key numbers

| Item | Value |
|---|---|
| Data | 80 train trips (208,417 GPS pts), 10 student-test trips (19,160 pts), 10 evaluator trips (hidden); one road network per trip |
| Candidates | 12 nearest arcs within 60 m; miss rate ≈ 0.6 % (nearest-arc-only would be wrong ≈ 44 %) |
| Features | 33 per (point, candidate) pair |
| Models compared | 10 (LogReg, GaussianNB, KNN, DecisionTree, RandomForest, ExtraTrees, HistGB, LightGBM, XGBoost, MLP) |
| Selection metric | 5-fold **group** CV Macro-F1 (also Top-1, log loss) |
| Best model | **XGBoost**: CV Macro-F1 0.9290; test Top-1 90.06 %, Macro-F1 0.9395, log loss 0.0628 |
| Bayesian tuning | Optuna TPE, 30 trials (23 complete, 7 pruned), ≈ 32 min. CV Macro-F1 0.9290 → 0.9299; test Top-1 90.06 % → 90.39 %, log loss 0.0628 → 0.0618 (small, honest gain) |
| HMM (tuned on out-of-fold train data) | Top-1 90.39 % → **95.84 %**; route F1 0.869 → **0.992**; continuity 0.780 → 0.911 (ceiling 0.932); sequence similarity 0.487 → 0.971 |
| HMM per trip | improves every one of the 10 trips; best +11.4 pp (trip 77), smallest +1.9 pp (trip 65, the weakest at 88.7 %) |
| Classifier error types | opposite-direction arc 3.9 %, sibling arc 2.1 %, other 1.9 %, adjacent arc 1.7 % |
| API speed | 0.04 – 0.8 s per trip (all 10 trips) |

## 2. Suggested 10-minute talk (8 slides) and who says what

| # | Slide | Speaker | Content |
|---|---|---|---|
| 1 | Problem | Daksha | GPS noise, map matching, our pipeline in one diagram (candidates → features → classifier → HMM → route). Two-laptop architecture. |
| 2 | Data & EDA | Darshan | Files; each trip has its own network; label construction by monotone alignment; data-quality findings; Folium EDA map. Figures: `eda_overview.png`, `eda_trajectories.png`. |
| 3 | Candidates & features | Harsh | Radius/K ablation (`candidate_miss.png`); the 7 feature groups and why. |
| 4 | Models | Harsha | 10 models table, CV Macro-F1 bar chart, overfitting check (`test_comparison.png`, `training_curves.png`). Why XGBoost. |
| 5 | Bayesian optimisation | Harsha | Search space, TPE, pruning, history plot (`bayes_opt.png`), tuned vs untuned (small gain — say so). |
| 6 | HMM + Viterbi | Darshan | States, emission, transition, gaps/unreachable, recurrence; classifier-only vs HMM bars (`hmm_vs_classifier.png`); where it didn't help. |
| 7 | **Live demo** | Harsh + Daksha | See section 3. |
| 8 | Limitations & takeaways | Daksha | See section 5. |

Every member must still be able to answer any question — see `docs/VIVA_QA.md`. Rehearse once with the roles swapped.

## 3. Live demo script (two laptops)

Before the evaluator arrives (do this once, then leave both running):
1. Both laptops on the same Wi-Fi/hotspot. Laptop 1: `ipconfig getifaddr en0` → note the IP.
2. **Laptop 1**: `export PATHSNAP_EVAL_DIR="<evaluator trips folder>"` then `uvicorn api.main:app --host 0.0.0.0 --port 8000`. Leave this terminal visible — it prints the request log.
3. **Laptop 2**: `export PATHSNAP_API="http://<laptop-1-ip>:8000"` then `streamlit run ui/app.py`.
4. From laptop 2 open `http://<laptop-1-ip>:8000/docs` once to prove connectivity. Run `python scripts/10_test_api.py http://<laptop-1-ip>:8000` if you want a one-shot check of all ten trips.

During the demo:
1. Evaluator opens the UI → ten trip buttons appear (loaded live from `GET /trips`).
2. Evaluator picks **any** trip → click **Run PathSnap**.
3. Point at **laptop 1's terminal**: `--> POST /predict`, `PREDICT trip=…`, `PIPELINE … features / classifier / lattice / viterbi` timings, `RESULT … Top-1 classifier-only=… HMM=…`. The guidelines require you to show these API logs.
4. On the UI: metric cards (point accuracy HMM vs classifier, route F1, continuity, server time), the map (blue GPS, orange classifier-only layer, red HMM route; toggle layers), and the comparison table.
5. Run a second and a third trip. Say: "the reference route never leaves laptop 1 — the prediction function doesn't even receive it; only metrics come back."

Take the required screenshots now (UI on laptop 2; API log in the terminal on laptop 1) and paste them into report §7 where the placeholder note is.

## 4. Pre-demo checklist

- [ ] `artifacts/models/pathsnap_bundle.joblib` present (the API loads only this).
- [ ] Evaluator trip folders contain `<id>.track/.nodes/.arcs/.route` and `PATHSNAP_EVAL_DIR` points at them. **Do not** run any training/analysis script on that folder.
- [ ] Laptop 1: `pip install -r requirements.txt`; macOS needs `brew install libomp`.
- [ ] Firewall allows incoming connections on port 8000 (macOS prompts the first time — click Allow).
- [ ] Laptop 2: `pip install -r ui/requirements.txt` only (no model needed there).
- [ ] The UI has internet access for the OpenStreetMap tiles (or the map background is blank but routes still show).
- [ ] Dry-run all ten trips once the night before.
- [ ] If the API port is busy: `uvicorn api.main:app --host 0.0.0.0 --port 8001` and change the UI's "API base URL" box.

## 5. Be honest about the limits (this scores well in a viva)

- The tuned model is only slightly better than the untuned one; the top four models are within about one fold-standard-deviation. The big gain comes from sequence decoding.
- Reference routes are not independently verified: some are partial or have breaks; point labels are derived by alignment, so accuracy is bounded by label noise (we excluded points > 50 m from their aligned arc).
- The tuned HMM emission weight sits near the lower edge of its search range; a wider search might help slightly.
- Remaining errors are mostly at junctions, very short arcs, and trips with GPS gaps (trip 65 is the weakest).
- Student-test scores are higher than CV because the student-test trips have nearly complete labels, whereas the training folds include noisier trips.

## 6. Submission checklist

- [ ] **GitHub (public)**: `cd` into this folder → `git init`, each member commits their own part (repository history is reviewed), push to a public repo. Do not commit the dataset or any secrets (`.gitignore` already excludes the venv, the big feature tables and the large baseline models; model weights for the final bundle are included, 13 MB).
- [ ] README renders and its commands work on a clean machine.
- [ ] **One consolidated PDF**: `reports/PathSnap_Technical_Report.pdf` (+ add the two demo screenshots to §7).
- [ ] Interactive Folium HTML in `maps/` and the generating script `scripts/06_make_maps.py` are committed.
- [ ] Re-run the pipeline scripts yourselves at least once so that you can honestly say you reproduced the results; read the report and rewrite anything you would not say in your own words.
- [ ] AI-assistance disclosure is in the README, the report's first page and `docs/AI_DISCLOSURE.md` — keep it.
