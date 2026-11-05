"""Classify request strings from the command line or stdin.

    python -m src.predict "id=1' OR 1=1--" "<svg onload=alert(1)>" "cheap flights"
    echo "q=hello world" | python -m src.predict -
    python -m src.predict --model tfidf_svm --window 48 "long text ... payload ... more text"

On first use the chosen model is trained on the full synthetic dataset and
cached in models/ (git-ignored). Inputs are only *read as text*; nothing is
decoded-and-executed, rendered or sent anywhere.
"""
from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from . import config
from .data_generation import generate_dataset
from .models import MODEL_NAMES, build_model, malicious_score
from .windowed import WindowedDetector


def load_model(key: str, retrain: bool = False):
    path = config.MODELS_DIR / f"{key}.joblib"
    if path.exists() and not retrain:
        return joblib.load(path)
    if config.DATASET_CSV.exists():
        df = pd.read_csv(config.DATASET_CSV, keep_default_na=False)
    else:
        df = generate_dataset()
    print(f"[predict] training {MODEL_NAMES[key]} on {len(df):,} rows ...", file=sys.stderr)
    model = build_model(key, canonical=True).fit(df["text"], df["label"])
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Classify strings as benign / sqli / xss.")
    ap.add_argument("texts", nargs="+", help="strings to classify, or '-' to read lines from stdin")
    ap.add_argument("--model", default="hybrid_lr", choices=list(MODEL_NAMES))
    ap.add_argument("--window", type=int, default=0,
                    help="sliding-window width (0 = whole string); defends against padding")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args(argv)

    texts = [l.rstrip("\n") for l in sys.stdin] if args.texts == ["-"] else args.texts
    model = load_model(args.model, args.retrain)
    det = WindowedDetector(model, args.window, max(args.window // 3, 1)) if args.window else model
    preds = det.predict(texts)
    scores = det.malicious_score(texts) if args.window else malicious_score(model, texts)
    for t, p, s in zip(texts, preds, scores):
        shown = t if len(t) <= 70 else t[:67] + "..."
        print(f"{p:7s}  score={s:+.3f}  {shown!r}")


if __name__ == "__main__":
    main()
