"""Adds API / stored-results screenshots + copies figures into testing_results_screenshots/ (real captured outputs only)."""
import _bootstrap  # noqa
import importlib.util, shutil, datetime, subprocess
from psnap.config import *
spec = importlib.util.spec_from_file_location("shots", Path(__file__).with_name("15_terminal_screenshots.py")); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
from pathlib import Path
stamp = datetime.datetime.now().strftime("%d %b %Y %H:%M"); OUT = S.OUT
def go(stem, cmd, title, text):
    (OUT / f"{stem}.txt").write_text(f"$ {cmd}\n{text}\n")
    lines = [l[:190] for l in text.splitlines()]; ch = [lines[i:i + S.LINES_PER_IMG] for i in range(0, len(lines), S.LINES_PER_IMG)]
    for k, c in enumerate(ch, 1): S.render(cmd, title, c, stem, k, len(ch), stamp)
    print("rendered", stem)
go("08_api_test_all_10_trips", "python scripts/10_test_api.py http://127.0.0.1:8000", "End-to-end API test: /health, /trips, /predict for all 10 trips", (LOGS / "api_test_output.txt").read_text())
log = (LOGS / "api.log").read_text().splitlines()
go("09_api_server_log", "uvicorn api.main:app --host 0.0.0.0 --port 8000   # server log on laptop 1", "API server log (each /predict request, per-stage timings, result)", "\n".join(l[:190] for l in log[:42]))
saved = subprocess.run(["python", str(Path(__file__).with_name("demo_train_models.py")), "--saved"], capture_output=True, text=True).stdout
go("10_stored_results_table", "python scripts/demo_train_models.py --saved", "All 10 models: 5-fold CV + student-test accuracy (stored results)", saved)
FIG = OUT / "figures"; FIG.mkdir(exist_ok=True)
for f in ["ui_screenshot", "map_00000052", "map_00000007", "map_00000056", "cv_macro_f1", "test_comparison", "training_curves", "bayes_opt", "hmm_vs_classifier", "confusion_matrices", "eda_overview", "eda_trajectories"]:
    if (FIGS / f"{f}.png").exists(): shutil.copy(FIGS / f"{f}.png", FIG / f"{f}.png")
print("copied figures:", len(list(FIG.glob('*.png'))))
