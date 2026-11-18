# Testing-results proof (terminal output of the real commands)

Each image is the real output of the command shown at the top of the image, produced by a full end-to-end run
(`bash run_all.sh`, 18 Nov 2026, 11:26 - 12:13). The matching raw text is saved next to every image (`.txt`).
The images are rendered from the captured output text - they are not OS screenshots of a Terminal window. To make
live screenshots yourself, run the same command and press Cmd+Shift+4.

| Image(s) | Command | What it proves |
|---|---|---|
| `01_data_exploration_p1.png` | `python scripts/01_eda.py` | Dataset inventory, 0 NaNs, GPS gaps, network components, candidate-radius ablation |
| `02_data_preprocessing_p1-3.png` | `python scripts/02_build_dataset.py` | Real preprocessing of all 80 train + 10 student-test trips: labels, candidates, 33 features, per-trip stats |
| `03_train_10_models_cv_p1-2.png` | `python scripts/03_train_compare.py` | 10 models trained: 5-fold group CV for each fold + final test result (Macro-F1, Top-1, log loss, fit/inference time) |
| `10_stored_results_table_p1.png` | `python scripts/demo_train_models.py --saved` | One table with CV and student-test accuracy of all 10 models + best model |
| `04_bayesian_optimisation_p1-3.png` | `python scripts/04_bayes_opt.py auto 30` | 30 Optuna trials, best parameters, tuned vs untuned accuracy |
| `05_hmm_viterbi_evaluation_p1-2.png` | `python scripts/05_hmm_eval.py` | HMM tuning; classifier-only vs classifier+HMM per trip |
| `06_error_analysis_p1.png` | `python scripts/11_error_analysis.py` | Types of classifier mistakes |
| `07_model_comparison_table_p1.png` | `python scripts/07_model_plots.py` | Final comparison table |
| `08_api_test_all_10_trips_p1.png` | `python scripts/10_test_api.py http://127.0.0.1:8000` | API works end to end for all 10 trips |
| `09_api_server_log_p1.png` | `uvicorn api.main:app ...` | API server log (request, pipeline timings, result per trip) |
| `figures/` | - | Charts, Folium map exports and the UI screenshot (real browser render) |

## Key results (student-test partition, 10 trips)
* Best model: **XGBoost** - CV Macro-F1 0.9290, test Top-1 accuracy 90.06 % (tuned: 0.9299 / 90.39 %).
* Classifier + HMM/Viterbi: Top-1 **95.84 %**, route F1 **0.992**.
