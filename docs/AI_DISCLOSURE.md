# AI-assistance disclosure

The event policy requires AI assistance to be disclosed and forbids fabricated results.

**What was AI-assisted.** The code in this repository (data pipeline, models, HMM/Viterbi decoder, API, UI, report generator)
was written with the help of an AI coding assistant (Claude Code, Anthropic) working inside the team's project folder.

**What was NOT fabricated.** Every metric, table and figure in the report is produced by the scripts in `scripts/` and stored in
`artifacts/results/` and `artifacts/figures/`. Nothing was typed in by hand; the report generator reads the result files.
The experiments (10 models, group cross-validation, Optuna runs, HMM tuning) were really executed; logs are in `artifacts/logs/`.

**Team responsibility.** Each team member must be able to explain every component (see `docs/PRESENTATION_GUIDE.md`, Viva section).
Before submission the team should re-run the scripts on their own machines, read the report critically, and edit the narrative
sections in their own words where they disagree with or want to extend the analysis.

**Data-use rules followed.** Evaluator-partition data is never read by any training/tuning/analysis script; models and decoder
parameters were selected using training trips (group CV / out-of-fold predictions) and student-test trips only.
