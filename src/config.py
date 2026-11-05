"""Central configuration for the SQLi/XSS payload-detection project.

Every random process in the project draws its seed from here so that
`python -m src.run_experiments` regenerates bit-identical data, metrics and
figures on any machine with the pinned library versions.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"          # git-ignored; written by src.predict
DATASET_CSV = DATA_DIR / "payloads.csv"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
N_BENIGN = 10_000
N_SQLI = 5_000
N_XSS = 5_000

LABELS = ["benign", "sqli", "xss"]          # fixed class order everywhere

# Obfuscation families. "Seen" families may appear in training data for the
# obfuscation-shift experiment; "held-out" families appear ONLY in its test
# set, simulating an attacker who uses an encoding trick the model never saw.
SEEN_OBFUSCATIONS = ["none", "case", "comment_ws", "url_encode",
                     "html_entity", "tag_slash"]
HELD_OUT_OBFUSCATIONS = ["double_url", "whitespace_alt", "versioned_comment",
                         "unicode_escape", "char_encoding", "benign_padding"]

# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.20         # stratified hold-out fraction for the main split
CV_FOLDS = 5             # stratified k-fold for variance estimates
