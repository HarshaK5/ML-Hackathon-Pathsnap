"""Central configuration: paths and constants (no secrets here)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # repository root
# Raw dataset root (contains train/ and student_test/). Override with PATHSNAP_DATA.
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", ROOT.parent / "Student folder"))
# Directory the API serves trips from. On the evaluation day this is pointed at the
# evaluator folder (10 trips). During development it defaults to the student_test split.
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", DATA_ROOT / "student_test"))

ART = ROOT / "artifacts"
MODELS = ART / "models"
RESULTS = ART / "results"
FIGS = ART / "figures"
LOGS = ART / "logs"
MAPS = ROOT / "maps"

SEED = 42

# --- candidate generation ---
CAND_RADIUS_M = 60.0      # search radius around each GPS point (chosen from candidate-miss ablation)
CAND_K = 12               # keep the K nearest arcs inside the radius
CAND_FALLBACK_K = 3       # if nothing in radius, take nearest K arcs at any distance
LABEL_MAX_DIST_M = 50.0   # a point is "label-valid" if its aligned reference arc is within this

# --- features ---
CTX_OFFSETS = (-3, -2, -1, 1, 2, 3)   # neighbouring GPS points used for temporal context
