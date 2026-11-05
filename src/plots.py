"""Matplotlib figures. Static PNGs, one consistent visual system.

Colour roles (validated palette, see README "Figures"):
  * class identity  - benign = blue, SQLi = orange, XSS = aqua (fixed order)
  * preprocessing   - raw = light neutral grey, canonicalised = near-black
  * model identity  - categorical slots 4-8 (yellow, magenta, green, violet,
                      red), disjoint from the class hues
  * magnitude       - single-hue blue ramp (confusion matrices)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # headless, deterministic rendering
import matplotlib.pyplot as plt           # noqa: E402
import matplotlib.ticker                  # noqa: E402,F401
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config                      # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CLASS_COLORS = {"benign": "#2a78d6", "sqli": "#eb6834", "xss": "#1baf7a"}
CLASS_TITLES = {"benign": "Benign", "sqli": "SQLi", "xss": "XSS"}
PREP_COLORS = {"raw": "#a3a19a", "canonical": "#2b2a28"}
MODEL_COLORS = {"tfidf_nb": "#eda100", "tfidf_lr": "#e87ba4", "tfidf_svm": "#008300",
                "hand_rf": "#6250d6", "hybrid_lr": "#e34948"}
MODEL_MARKERS = {"tfidf_nb": "v", "tfidf_lr": "s", "tfidf_svm": "o", "hand_rf": "^", "hybrid_lr": "D"}
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "svg.hashsalt": "seed", "path.simplify": True,
})


def _save(fig, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=160, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
def length_distribution(df: pd.DataFrame) -> Path:
    """Payload length per class: share of each class per log-spaced length bin."""
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    lengths = df["text"].str.len()
    # integer-aligned log bins avoid empty sub-character bins at short lengths
    bins = np.unique(np.round(np.logspace(0, np.log10(lengths.max() + 1), 36)))
    for lab in config.LABELS:
        v = lengths[df["label"] == lab]
        ax.hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), histtype="step",
                linewidth=2, color=CLASS_COLORS[lab],
                label=f"{CLASS_TITLES[lab]}  (median {int(v.median())}, n={len(v):,})")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks([2, 5, 10, 20, 50, 100, 200, 500])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("String length in characters (log scale)")
    ax.set_ylabel("Share of class")
    ax.set_title("Payload length distribution by class", loc="left", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    _grid(ax, "y")
    return _save(fig, "length_distribution.png")


def _draw_cm(ax, cm: np.ndarray, title: str, show_cbar=False):
    norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(norm, cmap=BLUE_RAMP, vmin=0, vmax=1)
    ticks = [CLASS_TITLES[l] for l in config.LABELS]
    ax.set_xticks(range(3), ticks)
    ax.set_yticks(range(3), ticks)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left", fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.1%}", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.55 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def confusion_matrix_single(cm: np.ndarray, title: str, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    im = _draw_cm(ax, cm, title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Row-normalised share")
    cb.outline.set_visible(False)
    return _save(fig, name)


def confusion_matrix_grid(cms: dict, titles: dict, name: str, suptitle: str) -> Path:
    keys = list(cms)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 3.4))
    for ax, k in zip(np.atleast_1d(axes), keys):
        _draw_cm(ax, cms[k], titles[k])
    fig.suptitle(suptitle, x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


def top_ngrams(ngr: pd.DataFrame, name: str = "top_ngrams.png") -> Path:
    """Three panels: the n-grams with the largest LR weight for each class."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.6))
    for ax, lab in zip(axes, config.LABELS):
        d = ngr[ngr["class"] == lab].sort_values("weight")
        labels = [" " + t.replace(" ", "␣") + " " for t in d["ngram"]]
        ax.barh(range(len(d)), d["weight"], color=CLASS_COLORS[lab], height=0.7)
        ax.set_yticks(range(len(d)), labels, fontfamily="monospace")
        ax.set_title(CLASS_TITLES[lab], loc="left")
        ax.set_xlabel("LogReg coefficient")
        _grid(ax, "x")
    fig.suptitle("Most indicative character n-grams per class (TF-IDF + LogReg, canonicalised; ␣ = space)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return _save(fig, name)


def feature_importance(imp: pd.DataFrame, name: str = "feature_importance.png") -> Path:
    d = imp.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6.4, 6.6))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"],
            color=MODEL_COLORS["hand_rf"], height=0.7,
            error_kw=dict(ecolor=INK_2, lw=1, capsize=2))
    ax.set_xlabel("Permutation importance (drop in macro-F1 on test set)")
    ax.set_title("Hand-crafted feature importance (Random Forest)", loc="left")
    _grid(ax, "x")
    return _save(fig, name)


def model_comparison(metrics: pd.DataFrame, model_titles: dict,
                     name: str = "model_comparison.png") -> Path:
    """Dot plot of macro-F1 per model, raw vs canonicalised, for both experiments."""
    exps = [("in_distribution", "In-distribution test split"),
            ("obfuscation_shift", "Obfuscation-shift test (unseen encodings)")]
    keys = list(model_titles)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, (exp, title) in zip(axes, exps):
        d = metrics[metrics["experiment"] == exp]
        for prep, marker in [("raw", "o"), ("canonical", "D")]:
            vals = [d[(d.model == k) & (d.preprocessing == prep)]["macro_f1"].iloc[0] for k in keys]
            ax.scatter(vals, range(len(keys)), s=64, marker=marker, color=PREP_COLORS[prep],
                       edgecolor=SURFACE, linewidth=2, zorder=3,
                       label="raw (lower-case only)" if prep == "raw" else "canonicalised")
        for i, k in enumerate(keys):
            r = d[(d.model == k) & (d.preprocessing == "raw")]["macro_f1"].iloc[0]
            c = d[(d.model == k) & (d.preprocessing == "canonical")]["macro_f1"].iloc[0]
            ax.plot([r, c], [i, i], color=GRID, lw=2, zorder=2)
            ax.annotate(f"{c:.4f}", (c, i), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK_2)
        ax.set_yticks(range(len(keys)), [model_titles[k] for k in keys])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Macro-F1 (3-class)")
        _grid(ax, "x")
        ax.set_ylim(-0.6, len(keys) - 0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.07), ncol=2)
    fig.tight_layout()
    return _save(fig, name)


def obfuscation_detection(obf: pd.DataFrame, title: str,
                          name: str = "obfuscation_detection.png") -> Path:
    """Detection rate per obfuscation family, raw vs canonicalised."""
    order = (obf[obf.preprocessing == "canonical"].sort_values("detection_rate")["obfuscation"].tolist())
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(order) + 1.2))
    h = 0.38
    for off, prep in [(h / 2, "raw"), (-h / 2, "canonical")]:
        d = obf[obf.preprocessing == prep].set_index("obfuscation").loc[order]
        ax.barh(np.arange(len(order)) + off, d["detection_rate"], height=h - 0.04,
                color=PREP_COLORS[prep],
                label="raw (lower-case only)" if prep == "raw" else "canonicalised")
        for i, v in enumerate(d["detection_rate"]):
            ax.text(v + 0.005, i + off, f"{v:.1%}", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0, 1.09)
    ax.set_xlabel("Detection rate (malicious predicted as SQLi or XSS)")
    ax.set_title(title, loc="left", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    _grid(ax, "x")
    return _save(fig, name)


def window_sweep(sweep: pd.DataFrame, model_titles: dict, name: str = "window_sweep.png") -> Path:
    """Effect of sliding-window width on shift-test macro-F1 and in-distribution FPR."""
    order = [w for w in sorted(sweep["window"].unique(), key=lambda v: (v == "whole", str(v).zfill(6)))]
    xpos = {w: i for i, w in enumerate(order)}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    panels = [("obfuscation_shift", "macro_f1", "Obfuscation-shift test: macro-F1"),
              ("in_distribution", "bin_fpr", "In-distribution test: benign false-positive rate")]
    for ax, (exp, col, title) in zip(axes, panels):
        d = sweep[sweep.experiment == exp]
        for k in model_titles:
            dk = d[d.model == k].set_index("window").loc[order]
            ax.plot([xpos[w] for w in order], dk[col], color=MODEL_COLORS[k], lw=2,
                    marker=MODEL_MARKERS[k], ms=7, mec=SURFACE, mew=1.5, label=model_titles[k])
        ax.set_xticks(range(len(order)), [f"{w}" if w != "whole" else "whole\nstring" for w in order])
        ax.set_xlabel("Window width in characters (stride = width / 3)")
        ax.set_title(title, loc="left")
        _grid(ax, "y")
        if col == "bin_fpr":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def family_holdout(fam: pd.DataFrame, model_titles: dict, name: str = "family_holdout.png") -> Path:
    """Detection rate of each attack family when it is entirely absent from training."""
    keys = [k for k in model_titles if k in fam.model.unique()]
    fam = fam.assign(tag=fam["label"].str.upper().str.replace("SQLI", "SQLi") + " / " + fam["family"])
    order = (fam.groupby("tag")["detection_rate"].mean().sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(order) + 1.3))
    h = 0.8 / len(keys)
    for j, k in enumerate(keys):
        d = fam[fam.model == k].set_index("tag").loc[order]
        off = ((len(keys) - 1) / 2 - j) * h          # first model drawn on top
        ax.barh(np.arange(len(order)) + off, d["detection_rate"], height=h - 0.04,
                color=MODEL_COLORS[k], label=model_titles[k])
        for i, v in enumerate(d["detection_rate"]):
            ax.text(v + 0.005, i + off, f"{v:.0%}", va="center", fontsize=6.5, color=INK_2)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Detection rate on the held-out family (flagged as any attack)")
    ax.set_title("Leave-one-attack-family-out generalisation", loc="left", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(keys), borderaxespad=0.2)
    _grid(ax, "x")
    return _save(fig, name)
