"""End-to-end reproducible pipeline: data -> models -> metrics -> figures.

    python -m src.run_experiments            # full run (~2-4 min on a laptop)
    python -m src.run_experiments --quick    # skip cross-validation

Experiments
-----------
E1  In-distribution: stratified 80/20 split of the full dataset.
E2  5-fold stratified cross-validation (variance of E1 estimates).
E3  Obfuscation shift: train on benign-train + malicious rows whose obfuscation
    is in SEEN_OBFUSCATIONS; test on benign-test + malicious rows using one of
    the HELD_OUT_OBFUSCATIONS (never seen in training).
E4  Sliding-window inference (defence against benign padding): the canonical
    E1 and E3 models re-scored with WindowedDetector for several widths.
E5  Leave-one-attack-family-out: for each of the 14 SQLi/XSS families, train
    without it and measure how much of it is still detected.
Each of E1 and E3 is run twice: with canonicalisation and with raw
(lower-case only) input, for every model.

Outputs (all under results/): metrics.csv, cv_metrics.csv, dataset_summary.csv,
obfuscation_detection.csv, window_sweep.csv, family_holdout.csv, top_ngrams.csv, feature_importance.csv,
misclassified_examples.csv, classification_report.txt, timing.csv, tables.md,
summary.json and figures/*.png.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config, evaluate, make_tables, plots
from .data_generation import generate_dataset
from .models import MODEL_NAMES, build_model, malicious_score
from .windowed import WindowedDetector

PREPS = {"canonical": True, "raw": False}
WINDOWS = [32, 48, 64, 96]                 # E4 widths; stride = width // 3
LOFO_MODELS = list(MODEL_NAMES)            # E5 models


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_or_generate() -> pd.DataFrame:
    """Always regenerate (deterministic) and verify against the committed CSV."""
    df = generate_dataset()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.DATASET_CSV.exists():
        old = pd.read_csv(config.DATASET_CSV, keep_default_na=False)
        same = len(old) == len(df) and (old["text"].astype(str).values == df["text"].values).all()
        print(f"[data] regenerated dataset {'matches' if same else 'DIFFERS FROM'} committed CSV")
    df.to_csv(config.DATASET_CSV, index=False)
    return df


def fit_eval(key, canonical, train, test):
    """Fit one pipeline and return (fitted model, predictions, metric dict, seconds)."""
    model = build_model(key, canonical=canonical)
    t0 = time.perf_counter()
    model.fit(train["text"], train["label"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(test["text"])
    pred_s = time.perf_counter() - t0
    score = malicious_score(model, test["text"])
    m = evaluate.multiclass_metrics(test["label"], pred)
    m.update(evaluate.binary_metrics(test["label"], pred, score, test["hard_benign"]))
    return model, pred, m, fit_s, pred_s


def score_predictions(test, pred, score) -> dict:
    m = evaluate.multiclass_metrics(test["label"], pred)
    m.update(evaluate.binary_metrics(test["label"], pred, score, test["hard_benign"]))
    return m


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["label", "family", "obfuscation", "context"]:
        g = df.groupby(["label", col]).size().reset_index(name="n") if col != "label" \
            else df.groupby("label").size().reset_index(name="n")
        g.insert(0, "dimension", col)
        if col == "label":
            g["value"] = g["label"]
        else:
            g = g.rename(columns={col: "value"})
        rows.append(g[["dimension", "label", "value", "n"]])
    lengths = df.assign(length=df["text"].str.len()).groupby("label")["length"]
    stats = lengths.describe(percentiles=[0.5]).reset_index()
    for _, r in stats.iterrows():
        for s in ["mean", "50%", "min", "max"]:
            rows.append(pd.DataFrame([{"dimension": f"length_{s}", "label": r["label"],
                                       "value": "", "n": round(r[s], 2)}]))
    hard = df[df.label == "benign"]["hard_benign"].sum()
    rows.append(pd.DataFrame([{"dimension": "hard_benign", "label": "benign", "value": True, "n": int(hard)}]))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(quick: bool = False) -> None:
    np.random.seed(config.SEED)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_or_generate()
    df["hard_benign"] = df["hard_benign"].astype(bool)
    print(f"[data] {len(df):,} rows: " + ", ".join(f"{k}={v:,}" for k, v in df.label.value_counts().items()))
    dataset_summary(df).to_csv(config.RESULTS_DIR / "dataset_summary.csv", index=False)
    plots.length_distribution(df)

    metric_rows, timing_rows, obf_rows = [], [], []
    fitted, preds = {}, {}

    # ---------------- E1: in-distribution split ---------------------------- #
    train, test = train_test_split(df, test_size=config.TEST_SIZE, stratify=df["label"],
                                   random_state=config.SEED)
    print(f"[E1] train={len(train):,} test={len(test):,}")
    for prep, canon in PREPS.items():
        for key in MODEL_NAMES:
            model, pred, m, fs, ps = fit_eval(key, canon, train, test)
            fitted[("E1", prep, key)], preds[("E1", prep, key)] = model, pred
            metric_rows.append({"experiment": "in_distribution", "preprocessing": prep,
                                "model": key, "model_name": MODEL_NAMES[key], **m})
            timing_rows.append({"experiment": "in_distribution", "preprocessing": prep, "model": key,
                                "fit_seconds": fs, "predict_us_per_sample": 1e6 * ps / len(test)})
            mal = test["label"] != "benign"
            g = evaluate.recall_by_group(test[mal], pred[mal.values], "obfuscation")
            obf_rows.append(g.assign(experiment="in_distribution", preprocessing=prep, model=key))
            print(f"  [E1] {prep:9s} {key:10s} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
                  f"FPR={m['bin_fpr']:.4f} hardFPR={m['hard_benign_fpr']:.4f}")

    # ---------------- E3: obfuscation shift -------------------------------- #
    ben = df[df.label == "benign"]
    ben_tr, ben_te = train_test_split(ben, test_size=config.TEST_SIZE, random_state=config.SEED)
    mal = df[df.label != "benign"]
    s_train = pd.concat([ben_tr, mal[mal.obfuscation.isin(config.SEEN_OBFUSCATIONS)]])
    s_test = pd.concat([ben_te, mal[mal.obfuscation.isin(config.HELD_OUT_OBFUSCATIONS)]])
    print(f"[E3] train={len(s_train):,} test={len(s_test):,} "
          f"(held-out malicious={int((s_test.label != 'benign').sum()):,})")
    for prep, canon in PREPS.items():
        for key in MODEL_NAMES:
            model, pred, m, fs, ps = fit_eval(key, canon, s_train, s_test)
            fitted[("E3", prep, key)], preds[("E3", prep, key)] = model, pred
            metric_rows.append({"experiment": "obfuscation_shift", "preprocessing": prep,
                                "model": key, "model_name": MODEL_NAMES[key], **m})
            timing_rows.append({"experiment": "obfuscation_shift", "preprocessing": prep, "model": key,
                                "fit_seconds": fs, "predict_us_per_sample": 1e6 * ps / len(s_test)})
            malmask = (s_test["label"] != "benign").values
            g = evaluate.recall_by_group(s_test[malmask], pred[malmask], "obfuscation")
            obf_rows.append(g.assign(experiment="obfuscation_shift", preprocessing=prep, model=key))
            print(f"  [E3] {prep:9s} {key:10s} acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} "
                  f"binRecall={m['bin_recall']:.4f} FPR={m['bin_fpr']:.4f}")

    metrics = pd.DataFrame(metric_rows)
    metrics.round(5).to_csv(config.RESULTS_DIR / "metrics.csv", index=False)
    pd.DataFrame(timing_rows).round(4).to_csv(config.RESULTS_DIR / "timing.csv", index=False)
    obf = pd.concat(obf_rows, ignore_index=True)
    obf = obf[["experiment", "preprocessing", "model", "obfuscation", "n", "exact_recall", "detection_rate"]]
    obf.round(5).to_csv(config.RESULTS_DIR / "obfuscation_detection.csv", index=False)

    # ---------------- E4: sliding-window inference -------------------------- #
    sweep_rows = []
    for exp, tag, tst in [("in_distribution", "E1", test), ("obfuscation_shift", "E3", s_test)]:
        for key in MODEL_NAMES:
            base = metrics[(metrics.experiment == exp) & (metrics.preprocessing == "canonical")
                           & (metrics.model == key)].iloc[0].to_dict()
            sweep_rows.append({**base, "window": "whole", "stride": ""})
            pad = (tst["obfuscation"] == "benign_padding").to_numpy()
            sweep_rows[-1]["padding_detection"] = float((preds[(tag, "canonical", key)][pad] != "benign").mean()) if pad.any() else np.nan
            for w in WINDOWS:
                det = WindowedDetector(fitted[(tag, "canonical", key)], width=w, stride=w // 3)
                pred = det.predict(tst["text"])
                m = score_predictions(tst, pred, det.malicious_score(tst["text"]))
                m["padding_detection"] = float((pred[pad] != "benign").mean()) if pad.any() else np.nan
                sweep_rows.append({"experiment": exp, "preprocessing": "canonical", "model": key,
                                   "model_name": MODEL_NAMES[key], **m, "window": str(w), "stride": w // 3})
                if exp == "obfuscation_shift" and w == 48:
                    preds[("E4", "canonical", key)] = pred
        print(f"  [E4] {exp} sweep done")
    sweep = pd.DataFrame(sweep_rows)
    lead = ["experiment", "model", "model_name", "window", "stride"]
    sweep = sweep[lead + [c for c in sweep.columns if c not in lead and c != "preprocessing"]]
    sweep.round(5).to_csv(config.RESULTS_DIR / "window_sweep.csv", index=False)
    plots.window_sweep(sweep, MODEL_NAMES)

    # ---------------- E5: leave-one-attack-family-out ----------------------- #
    fam_rows = []
    mal_tr = df.loc[train.index][df.loc[train.index, "label"] != "benign"]
    ben_tr_e1, ben_te_e1 = train[train.label == "benign"], test[test.label == "benign"]
    for lab in ["sqli", "xss"]:
        for fam in sorted(df[df.label == lab]["family"].unique()):
            held = df[(df.label == lab) & (df.family == fam)]
            tr = pd.concat([ben_tr_e1, mal_tr[~((mal_tr.label == lab) & (mal_tr.family == fam))]])
            for key in LOFO_MODELS:
                model = build_model(key, canonical=True).fit(tr["text"], tr["label"])
                p_held = model.predict(held["text"])
                p_ben = model.predict(ben_te_e1["text"])
                fam_rows.append({"model": key, "model_name": MODEL_NAMES[key], "label": lab, "family": fam,
                                 "n": len(held), "detection_rate": float((p_held != "benign").mean()),
                                 "exact_recall": float((p_held == lab).mean()),
                                 "benign_fpr": float((p_ben != "benign").mean())})
            print(f"  [E5] {lab}/{fam} done")
    fam_df = pd.DataFrame(fam_rows)
    fam_df.round(5).to_csv(config.RESULTS_DIR / "family_holdout.csv", index=False)
    plots.family_holdout(fam_df, {k: MODEL_NAMES[k] for k in ["tfidf_svm", "hand_rf", "hybrid_lr"]})

    # ---------------- E2: cross-validation ---------------------------------- #
    if not quick:
        skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
        cv_rows = []
        for key in MODEL_NAMES:
            for fold, (tr, te) in enumerate(skf.split(df["text"], df["label"])):
                _, _, m, _, _ = fit_eval(key, True, df.iloc[tr], df.iloc[te])
                cv_rows.append({"model": key, "fold": fold, **m})
            print(f"  [E2] {key} done")
        cv = pd.DataFrame(cv_rows)
        cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "bin_f1", "bin_fpr", "roc_auc"]
        agg = cv.groupby("model")[cols].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg = agg.reindex(list(MODEL_NAMES)).reset_index()
        agg.insert(1, "model_name", agg["model"].map(MODEL_NAMES))
        agg.round(5).to_csv(config.RESULTS_DIR / "cv_metrics.csv", index=False)
        cv.round(5).to_csv(config.RESULTS_DIR / "cv_folds.csv", index=False)

    # ---------------- Best model artefacts ---------------------------------- #
    e1c = metrics[(metrics.experiment == "in_distribution") & (metrics.preprocessing == "canonical")]
    best = e1c.sort_values(["macro_f1", "bin_fpr"], ascending=[False, True]).iloc[0]["model"]
    best_pred = preds[("E1", "canonical", best)]
    print(f"[best] {best} ({MODEL_NAMES[best]})")

    with open(config.RESULTS_DIR / "classification_report.txt", "w") as fh:
        for prep in PREPS:
            for key in MODEL_NAMES:
                fh.write(f"=== E1 in-distribution | {prep} | {MODEL_NAMES[key]} ===\n")
                fh.write(classification_report(test["label"], preds[("E1", prep, key)],
                                               labels=config.LABELS, digits=4))
                fh.write("\n")

    # confusion matrices
    cms = {k: evaluate.confusion(test["label"], preds[("E1", "canonical", k)]) for k in MODEL_NAMES}
    cm_json = {f"E1_canonical_{k}": v.tolist() for k, v in cms.items()}
    cm_json.update({f"E3_{p}_{k}": evaluate.confusion(s_test["label"], preds[("E3", p, k)]).tolist()
                    for p in PREPS for k in MODEL_NAMES})
    with open(config.RESULTS_DIR / "confusion_matrices.json", "w") as fh:
        json.dump({"labels": config.LABELS, **cm_json}, fh, indent=1)
    plots.confusion_matrix_single(cms[best], f"{MODEL_NAMES[best]} - in-distribution test (n={len(test):,})",
                                  "confusion_matrix_best.png")
    plots.confusion_matrix_grid(cms, MODEL_NAMES, "confusion_matrices_all.png",
                                "Confusion matrices, in-distribution test split, canonicalised input")
    shift_cms = {f"{p}": evaluate.confusion(s_test["label"], preds[("E3", p, best)]) for p in ["raw", "canonical"]}
    plots.confusion_matrix_grid(shift_cms, {"raw": f"{MODEL_NAMES[best]} - raw",
                                            "canonical": f"{MODEL_NAMES[best]} - canonicalised"},
                                "confusion_matrix_shift.png",
                                "Obfuscation-shift test: unseen encodings only")

    # top n-grams from the TF-IDF + LogReg model
    lr = fitted[("E1", "canonical", "tfidf_lr")]
    vocab = lr.named_steps["tfidf"].get_feature_names_out()
    coef = lr.named_steps["clf"].coef_
    rows = []
    for ci, lab in enumerate(lr.named_steps["clf"].classes_):
        for idx in np.argsort(coef[ci])[::-1][:15]:
            rows.append({"class": lab, "ngram": vocab[idx], "weight": float(coef[ci, idx])})
    ngr = pd.DataFrame(rows)
    ngr.round(5).to_csv(config.RESULTS_DIR / "top_ngrams.csv", index=False)
    plots.top_ngrams(ngr)

    # permutation importance for the hand-crafted RF (computed on features
    # directly so each permutation does not re-extract them)
    rf_pipe = fitted[("E1", "canonical", "hand_rf")]
    Xte = rf_pipe.named_steps["hand"].transform(test["text"])
    pi = permutation_importance(rf_pipe.named_steps["clf"], Xte, test["label"], n_repeats=5,
                                random_state=config.SEED, n_jobs=1,
                                scoring=make_scorer(f1_score, average="macro"))
    names = rf_pipe.named_steps["hand"].feature_names_
    imp = pd.DataFrame({"feature": names, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std,
                        "gini_importance": rf_pipe.named_steps["clf"].feature_importances_})
    imp.sort_values("importance_mean", ascending=False).round(5).to_csv(
        config.RESULTS_DIR / "feature_importance.csv", index=False)
    plots.feature_importance(imp)

    # misclassified examples (best model, E1 canonical) for qualitative analysis
    wrong = test.assign(predicted=best_pred)
    wrong = wrong[wrong["predicted"] != wrong["label"]]
    wrong[["id", "label", "predicted", "family", "obfuscation", "hard_benign", "text"]].to_csv(
        config.RESULTS_DIR / "misclassified_examples.csv", index=False)
    # and the best model's misses under obfuscation shift (canonical)
    sp = s_test.assign(predicted=preds[("E3", "canonical", best)])
    sp = sp[sp["predicted"] != sp["label"]]
    sp[["id", "label", "predicted", "family", "obfuscation", "hard_benign", "text"]].to_csv(
        config.RESULTS_DIR / "misclassified_shift_examples.csv", index=False)

    plots.model_comparison(metrics, MODEL_NAMES)
    o = obf[(obf.experiment == "obfuscation_shift") & (obf.model == best)]
    plots.obfuscation_detection(o, f"Held-out obfuscations: detection rate, {MODEL_NAMES[best]}")
    shift_best = (metrics[(metrics.experiment == "obfuscation_shift") & (metrics.preprocessing == "canonical")]
                  .sort_values("macro_f1", ascending=False).iloc[0]["model"])

    summary = {
        "n_rows": int(len(df)), "class_counts": df.label.value_counts().to_dict(),
        "best_model": best, "best_model_name": MODEL_NAMES[best],
        "e1_test_size": int(len(test)), "e3_train_size": int(len(s_train)), "e3_test_size": int(len(s_test)),
        "e3_best_model": shift_best,
        "e3_canonical": metrics[(metrics.experiment == "obfuscation_shift") & (metrics.preprocessing == "canonical")]
        .set_index("model")[["accuracy", "macro_f1", "bin_recall", "bin_fpr"]].round(5).to_dict("index"),
        "e4_window48_shift": sweep[(sweep.experiment == "obfuscation_shift") & (sweep.window == "48")]
        .set_index("model")[["accuracy", "macro_f1", "bin_recall", "bin_fpr", "padding_detection"]].round(5).to_dict("index"),
        "e5_mean_detection": fam_df.groupby("model")["detection_rate"].mean().round(5).to_dict(),
        "e1_canonical": e1c.set_index("model")[["accuracy", "macro_f1", "bin_fpr", "roc_auc"]].round(5).to_dict("index"),
    }
    with open(config.RESULTS_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    make_tables.main()
    print("[done] results written to results/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip 5-fold cross-validation")
    main(**vars(ap.parse_args()))
