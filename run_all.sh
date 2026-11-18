#!/usr/bin/env bash
# Runs the ENTIRE PathSnap pipeline end to end (about 1 hour): EDA -> preprocessing -> 10 models (5-fold group CV)
# -> Bayesian optimisation -> HMM tuning/evaluation -> plots/maps -> technical report.
# Usage:  bash run_all.sh        (set PATHSNAP_DATA first if the dataset is not next to this folder)
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
step() { echo; echo "################ $1   ($(date +%H:%M:%S))"; }
step "1/11  EDA + data-quality checks";                 python scripts/01_eda.py
step "2/11  Data preprocessing (labels, candidates, 33 features)"; python scripts/02_build_dataset.py
step "3/11  Train + compare 10 models (5-fold group CV)"; rm -f artifacts/results/models/*.json; python scripts/03_train_compare.py
step "4/11  Bayesian hyper-parameter optimisation";     python scripts/04_bayes_opt.py auto 30
step "5/11  HMM / Viterbi tuning + evaluation";          python scripts/05_hmm_eval.py
step "6/11  Error analysis";                             python scripts/11_error_analysis.py
step "7/11  Training curves";                            python scripts/08_training_curves.py 2>/dev/null
step "8/11  Model comparison tables + figures";          python scripts/07_model_plots.py
step "9/11  Continuity ceiling";                         python scripts/13_reference_continuity.py
step "10/11 Folium maps";                                python scripts/06_make_maps.py
step "11/11 Technical report PDF";                       python scripts/09_make_report.py
echo; echo "################ ALL DONE ($(date +%H:%M:%S)).  Start the system:  uvicorn api.main:app --host 0.0.0.0 --port 8000   +   streamlit run ui/app.py"
