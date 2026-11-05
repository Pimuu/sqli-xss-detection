"""Render the Markdown tables used in paper/paper.md from results/*.csv.

    python -m src.make_tables      # writes results/tables.md

Keeping this step scripted means every number in the paper can be traced back
to a CSV produced by `src.run_experiments`.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .models import MODEL_NAMES

R = config.RESULTS_DIR


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def f4(x):
    return f"{x:.4f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


def dataset_table() -> str:
    df = pd.read_csv(config.DATASET_CSV, keep_default_na=False)
    rows = []
    for lab in config.LABELS:
        d = df[df.label == lab]
        L = d.text.str.len()
        rows.append({"Class": lab, "n": f"{len(d):,}", "families": d.family.nunique(),
                     "plain (no obfuscation)": pct((d.obfuscation == "none").mean()),
                     "seen-obf.": pct(d.obfuscation.isin(config.SEEN_OBFUSCATIONS[1:]).mean()),
                     "held-out-obf.": pct(d.obfuscation.isin(config.HELD_OUT_OBFUSCATIONS).mean()),
                     "param context": pct((d.context != "raw").mean()),
                     "median len": int(L.median()), "mean len": f"{L.mean():.1f}", "max len": int(L.max())})
    return _md(pd.DataFrame(rows))


def main_table(exp: str) -> str:
    m = pd.read_csv(R / "metrics.csv")
    rows = []
    for prep in ["canonical", "raw"]:
        for k in MODEL_NAMES:
            r = m[(m.experiment == exp) & (m.preprocessing == prep) & (m.model == k)].iloc[0]
            rows.append({"Input": prep, "Model": MODEL_NAMES[k], "Accuracy": f4(r.accuracy),
                         "Macro-P": f4(r.macro_precision), "Macro-R": f4(r.macro_recall),
                         "Macro-F1": f4(r.macro_f1), "F1 benign": f4(r.benign_f1),
                         "F1 SQLi": f4(r.sqli_f1), "F1 XSS": f4(r.xss_f1),
                         "Bin. recall": f4(r.bin_recall), "FPR": pct(r.bin_fpr, 2),
                         "Hard-benign FPR": pct(r.hard_benign_fpr, 2), "ROC-AUC": f4(r.roc_auc)})
    return _md(pd.DataFrame(rows))


def cv_table() -> str:
    c = pd.read_csv(R / "cv_metrics.csv")
    rows = [{"Model": r.model_name,
             "Accuracy": f"{r.accuracy_mean:.4f} ± {r.accuracy_std:.4f}",
             "Macro-F1": f"{r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f}",
             "Binary F1": f"{r.bin_f1_mean:.4f} ± {r.bin_f1_std:.4f}",
             "FPR": f"{100 * r.bin_fpr_mean:.2f} ± {100 * r.bin_fpr_std:.2f} %",
             "ROC-AUC": f"{r.roc_auc_mean:.5f}"} for r in c.itertuples()]
    return _md(pd.DataFrame(rows))


def obfuscation_table() -> str:
    o = pd.read_csv(R / "obfuscation_detection.csv")
    o = o[o.experiment == "obfuscation_shift"]
    piv = o.pivot_table(index="obfuscation", columns=["model", "preprocessing"], values="detection_rate")
    n = o.groupby("obfuscation")["n"].first()
    rows = []
    for ob in config.HELD_OUT_OBFUSCATIONS:
        row = {"Held-out obfuscation": ob, "n": int(n[ob])}
        for k in MODEL_NAMES:
            row[MODEL_NAMES[k]] = f"{pct(piv.loc[ob, (k, 'raw')])} → {pct(piv.loc[ob, (k, 'canonical')])}"
        rows.append(row)
    return _md(pd.DataFrame(rows))


def window_table() -> str:
    w = pd.read_csv(R / "window_sweep.csv", dtype={"window": str})
    rows = []
    for k in MODEL_NAMES:
        for win in ["whole", "32", "48", "64", "96"]:
            s = w[(w.experiment == "obfuscation_shift") & (w.model == k) & (w.window == win)].iloc[0]
            i = w[(w.experiment == "in_distribution") & (w.model == k) & (w.window == win)].iloc[0]
            rows.append({"Model": MODEL_NAMES[k], "Window": win,
                         "E3 macro-F1": f4(s.macro_f1), "E3 bin. recall": f4(s.bin_recall),
                         "E3 padding detected": pct(s.padding_detection), "E3 FPR": pct(s.bin_fpr, 2),
                         "E1 macro-F1": f4(i.macro_f1), "E1 FPR": pct(i.bin_fpr, 2)})
    return _md(pd.DataFrame(rows))


def family_table() -> str:
    f = pd.read_csv(R / "family_holdout.csv")
    piv = f.pivot_table(index=["label", "family"], columns="model", values="detection_rate")
    n = f.groupby(["label", "family"])["n"].first()
    rows = []
    for (lab, fam), r in piv.iterrows():
        rows.append({"Held-out family": f"{lab} / {fam}", "n": int(n[(lab, fam)]),
                     **{MODEL_NAMES[k]: pct(r[k]) for k in MODEL_NAMES}})
    mean = f.groupby("model")["detection_rate"].mean()
    fpr = f.groupby("model")["benign_fpr"].mean()
    rows.append({"Held-out family": "**mean detection**", "n": "",
                 **{MODEL_NAMES[k]: f"**{pct(mean[k])}**" for k in MODEL_NAMES}})
    rows.append({"Held-out family": "mean benign FPR", "n": "",
                 **{MODEL_NAMES[k]: pct(fpr[k], 2) for k in MODEL_NAMES}})
    return _md(pd.DataFrame(rows))


def importance_table(top: int = 10) -> str:
    imp = pd.read_csv(R / "feature_importance.csv").head(top)
    imp = imp.assign(perm=lambda d: d.importance_mean.map(f4) + " ± " + d.importance_std.map(f4),
                     gini=lambda d: d.gini_importance.map(f4))
    return _md(imp[["feature", "perm", "gini"]].rename(columns={
        "feature": "Feature", "perm": "Permutation Δ macro-F1", "gini": "Gini importance"}))


def ngram_table(top: int = 8) -> str:
    g = pd.read_csv(R / "top_ngrams.csv", keep_default_na=False)
    rows = []
    for lab in config.LABELS:
        d = g[g["class"] == lab].head(top)
        rows.append({"Class": lab, "Top n-grams (LogReg weight)":
                     ", ".join(f"`{t.replace(' ', '␣').replace('|', '¦')}` ({w:.1f})" for t, w in zip(d.ngram, d.weight))})
    return _md(pd.DataFrame(rows))


def timing_table() -> str:
    t = pd.read_csv(R / "timing.csv")
    t = t[(t.experiment == "in_distribution") & (t.preprocessing == "canonical")]
    return _md(pd.DataFrame([{"Model": MODEL_NAMES[r.model], "Fit time (s, 16k rows)": f"{r.fit_seconds:.1f}",
                              "Predict (µs / string)": f"{r.predict_us_per_sample:.0f}"} for r in t.itertuples()]))


def main() -> None:
    parts = {
        "Dataset": dataset_table(),
        "E1 in-distribution": main_table("in_distribution"),
        "E2 cross-validation": cv_table(),
        "E3 obfuscation shift": main_table("obfuscation_shift"),
        "E3 per-obfuscation detection (raw → canonical)": obfuscation_table(),
        "E4 sliding window": window_table(),
        "E5 leave-one-family-out": family_table(),
        "Feature importance": importance_table(),
        "Top n-grams": ngram_table(),
    }
    out = "\n\n".join(f"## {k}\n\n{v}" for k, v in parts.items()) + "\n"
    (R / "tables.md").write_text(out)
    print(out)
    # timing is machine-dependent, so it is printed but kept out of tables.md
    print("## Timing (machine-dependent, not saved)\n\n" + timing_table())


if __name__ == "__main__":
    main()
