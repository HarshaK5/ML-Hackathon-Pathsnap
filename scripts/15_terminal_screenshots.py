"""Render the REAL output of the pipeline commands (captured in artifacts/logs/run_all.log) as terminal-style PNG images
in testing_results_screenshots/, next to the raw text. Nothing is typed in by hand: images are drawn from the log text.
Usage: python scripts/15_terminal_screenshots.py"""
import _bootstrap  # noqa
import html, re, subprocess, datetime
from pathlib import Path
from psnap.config import ROOT, LOGS

OUT = ROOT / "testing_results_screenshots"; OUT.mkdir(exist_ok=True)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LINES_PER_IMG = 44
STEPS = {  # marker number -> (file stem, command shown, title)
    1: ("01_data_exploration", "python scripts/01_eda.py", "Data exploration + data-quality checks + candidate-radius ablation"),
    2: ("02_data_preprocessing", "python scripts/02_build_dataset.py", "Data preprocessing: labels, candidate arcs, 33 features (all 90 trips)"),
    3: ("03_train_10_models_cv", "python scripts/03_train_compare.py", "Training + testing 10 models (5-fold group CV, then student-test evaluation)"),
    4: ("04_bayesian_optimisation", "python scripts/04_bayes_opt.py auto 30", "Bayesian hyper-parameter optimisation (Optuna TPE)"),
    5: ("05_hmm_viterbi_evaluation", "python scripts/05_hmm_eval.py", "HMM/Viterbi tuning and classifier-only vs HMM evaluation"),
    6: ("06_error_analysis", "python scripts/11_error_analysis.py", "Error analysis of classifier-only predictions"),
    8: ("07_model_comparison_table", "python scripts/07_model_plots.py", "Final model comparison table"),
}


def sections(log):
    parts = re.split(r"^################ ", log, flags=re.M)[1:]
    out = {}
    for i, p in enumerate(parts):
        head, _, body = p.partition("\n")
        m = re.match(r"(\d+)/11", head)
        if m and i + 1 < len(parts):                      # complete only if a following marker exists
            out[int(m.group(1))] = (head.strip(), body.strip("\n"))
    return out


def render(cmd, title, lines, stem, part, nparts, stamp):
    esc = "\n".join(html.escape(l) for l in lines)
    esc = re.sub(r"(\d+\.\d+%|Top-1[^|]*|macroF1=\d\.\d+|CV macroF1 [\d.±]+|TEST macroF1 [\d.]+ top1 [\d.]+)", r'<b class="hl">\1</b>', esc)
    wchars = max([len(l) for l in lines] + [len(cmd) + 2, 60])
    W = min(max(int(wchars * 7.55) + 60, 900), 1700); H = int(len(lines) * 17.6) + 150
    doc = f"""<html><body style="margin:0;background:#2b2b2b"><div style="background:#1e1e1e;border-radius:10px;overflow:hidden;font-family:Menlo,Monaco,monospace">
<div style="background:#3c3c3c;padding:9px 14px;color:#ddd;font:13px -apple-system,Helvetica;display:flex;gap:8px;align-items:center">
<span style="width:12px;height:12px;border-radius:6px;background:#ff5f56"></span><span style="width:12px;height:12px;border-radius:6px;background:#ffbd2e"></span><span style="width:12px;height:12px;border-radius:6px;background:#27c93f"></span>
<span style="margin-left:12px">{html.escape(title)} &mdash; part {part}/{nparts}</span></div>
<pre style="margin:0;padding:12px 16px;color:#d4d4d4;font-size:12.2px;line-height:17.6px"><span style="color:#6a9955"># PathSnap/pathsnap | captured {stamp} | output of the real command</span>
<span style="color:#4ec9b0">$</span> <span style="color:#dcdcaa">{html.escape(cmd)}</span>
{esc}</pre></div><style>.hl{{color:#ffd866}}</style></body></html>"""
    h = OUT / f"{stem}_p{part}.html"; h.write_text(doc)
    png = OUT / f"{stem}_p{part}.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}", f"--screenshot={png}", h.resolve().as_uri()], capture_output=True, timeout=90)
    h.unlink(); return png


def main():
    log = (LOGS / "run_all.log").read_text(); stamp = datetime.datetime.now().strftime("%d %b %Y %H:%M")
    sec = sections(log); made = []
    for n, (stem, cmd, title) in STEPS.items():
        if n not in sec: continue
        head, body = sec[n]
        (OUT / f"{stem}.txt").write_text(f"$ {cmd}\n{body}\n")
        lines = body.splitlines(); lines = [l[:190] for l in lines]
        chunks = [lines[i:i + LINES_PER_IMG] for i in range(0, len(lines), LINES_PER_IMG)] or [[""]]
        for k, ch in enumerate(chunks, 1):
            made.append(render(cmd, title, ch, stem, k, len(chunks), stamp))
        print("rendered", stem, f"({len(chunks)} image(s))")
    return made


if __name__ == "__main__":
    main()
