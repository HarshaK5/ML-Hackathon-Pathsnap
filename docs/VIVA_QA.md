# Viva preparation — questions the organisers listed, with answers tied to OUR implementation

The policy says every member must be able to explain: dataset structure, preprocessing, feature engineering, model
comparison and selection, Bayesian optimisation, the HMM, Viterbi, the API and the UI. Read these, then open the code
file named in brackets and trace it once. Do not memorise — understand.

## Dataset and preprocessing
**What is in a trip folder?** `.track` (lon, lat, time), `.nodes` (lon, lat; row = node id), `.arcs` (from, to; row = arc id, directed),
`.route` (ordered arc ids of the reference route). Each trip has its OWN road network. [src/psnap/data.py]

**Why not use .route rows against .track rows?** Different lengths (e.g. 1,045 GPS points vs 105 route arcs). The route says which arcs, in order, but not
which point is on which arc. [Dataset PDF §3 warning]

**How did you get a label for every GPS point?** Monotone dynamic-programming alignment of points to the route arcs: point *i* gets route
position *j(i)*, never decreasing, minimising summed squared point-to-arc distance. Points still >50 m from their arc are "label-invalid" and dropped. [src/psnap/labeling.py]

**Why project to metres?** Degrees are angular; a degree of longitude shrinks with cos(latitude). We use a local azimuthal-equidistant projection per trip (pyproj). [data.py `make_projector`]

**Data-quality issues you found?** Large GPS gaps in some trips; some trips whose track is much longer than the reference route / extends past the extracted network
(e.g. train 00000054); small breaks and repeated arcs in some reference routes; networks with many disconnected components. Table in report §2.3.

**How did you avoid leakage?** Splits are by trip; CV uses GroupKFold on trips; features are computed per trip with no fitted preprocessing; the evaluator partition is never read;
the prediction function never receives the route. [scripts/03, evaluation.py]

## Candidates and features
**Why candidates?** A road network has up to hundreds of thousands of arcs; we only score arcs near each GPS point. Radius 60 m and K=12 came from the miss-rate ablation (report table 2).

**Why is "nearest arc" not enough?** Two-way streets give two arcs with identical geometry; nearest-arc picks the right one only about half the time. Direction needs heading/context features and the HMM.

**Name your feature groups.** Geometric fit (distance, side, rank, margins), position on arc, heading agreement, topology (degree, two-way), time gaps, temporal context
(neighbouring points' distance to the same arc), motion along the arc. [src/psnap/features.py]

**Why temporal-context features?** A true arc stays close for consecutive fixes; a crossing road is only close for one or two points.

## Models
**Which models and why?** 10 models, 8 families: LogReg, GaussianNB, KNN, DecisionTree, RandomForest, ExtraTrees, HistGB, LightGBM, XGBoost, MLP. Report §3.2 has a subsection each.

**Primary metric?** CV Macro-F1 of the candidate classification (class imbalance ≈ 12.9 % positives), with Top-1 accuracy and log loss alongside. Log loss matters because the probabilities are the HMM's emissions.

**How did you detect over/underfitting?** Train (20 k-point sample) vs CV vs student-test Macro-F1 and fold-to-fold std; learning curves for iterative models (report Fig. 8).

**Why is student-test higher than CV?** The student-test trips have almost complete labels (99.9 % valid) while the training folds contain noisier trips with partial reference routes.

**Why did tree ensembles/boosting beat logistic regression?** Non-linear interactions (distance matters differently when heading is unreliable or at junctions).

**Why is GaussianNB worst?** Features are strongly correlated (the context distances) and not Gaussian, so the independence assumption fails; see its high log loss.

## Bayesian optimisation
**What is it?** Optuna TPE: a surrogate model of "which hyper-parameter regions gave good scores" proposes the next trial instead of random/grid search. Objective: mean CV Macro-F1 over 5 group folds; MedianPruner stops bad trials early;
first trial = the untuned config. Search record in `artifacts/results/bayes_trials.csv` and `optuna_study.db`. [scripts/04_bayes_opt.py]

**Did tuning help?** See report §4 — report the actual difference honestly, including if it is within fold noise.

## HMM and Viterbi
**Hidden states?** Candidate arcs at each GPS step (the same candidates the classifier scored).

**Emission?** λ · log(classifier probability of the candidate).

**Transition?** −|route distance − GPS distance| / β, where route distance comes from shortest paths on the road graph; unreachable pairs get a fixed penalty; U-turns an extra one; after long gaps only reachability matters.

**Viterbi recurrence?** δ_t(b) = max_a [δ_{t−1}(a) + log T(a→b)] + emission_t(b); store argmax back-pointers; backtrack from the best final state. Complexity O(N·K²).

**Disconnected segments / gaps / no candidates?** Fixed 'route-break' penalty so decoding continues; reachability-only transitions after gaps; candidate fallback to the nearest arcs at any distance; shortest-path bridging of disconnected consecutive arcs after decoding. [src/psnap/hmm.py]

**Why precompute a "lattice"?** The route distances don't depend on the HMM parameters, so tuning the parameters is cheap.

**How were decoder parameters tuned without leakage?** On out-of-fold classifier probabilities of 30 training trips; the classifier had never seen those trips. Student-test only for reporting.

**Where does the HMM not help?** See report table 5 and the "did not help" sentence — usually junctions, very short arcs, GPS gaps.

## API and UI
**What does /predict do?** Loads the raw track and network → candidates → features → model → lattice → Viterbi → completed route; only then reads the reference route and computes metrics; returns polylines + metrics + timings. [api/main.py, pipeline.py, evaluation.py]

**How do you prove no ground truth leaks?** `pipeline.predict` has no route argument; `evaluation.evaluate` is called afterwards on the server; the UI only receives predictions and metrics.

**What do the API logs show?** Request in/out lines with status and latency, then PIPELINE (per-stage times) and RESULT (accuracy) lines for each trip. File: `artifacts/logs/api.log`.

**How do the two laptops talk?** HTTP over the same network: UI → `POST http://<laptop-1-ip>:8000/predict`.
