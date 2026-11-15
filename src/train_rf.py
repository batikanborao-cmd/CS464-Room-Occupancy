"""Progress-report baseline: Random Forest with default hyperparameters on the fixed day-based split.

Main metric: macro-F1 (TA feedback). Accuracy reported as an additional score.
No hyperparameter tuning yet; when tuning starts it must use the validation split only.
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from common import FIG_DIR, LABELS, RES_DIR, TARGET, feature_columns, get_split, load_data

SEED = 42


def evaluate(model, X, y):
    pred = model.predict(X)
    present = sorted(set(y))  # validation day has no class-1 samples
    return {
        "macro_f1": f1_score(y, pred, labels=present, average="macro", zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "classes_in_macro_f1": [int(c) for c in present],
        "per_class_f1": dict(zip(map(str, LABELS), f1_score(y, pred, labels=LABELS, average=None, zero_division=0).round(4).tolist())),
    }, pred


def main():
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    df = load_data()
    feats = feature_columns(df)
    split = get_split(df)
    X = {k: v[feats] for k, v in split.items()}
    y = {k: v[TARGET].to_numpy() for k, v in split.items()}

    model = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    t0 = time.perf_counter()
    model.fit(X["train"], y["train"])
    train_time = time.perf_counter() - t0

    results = {"model": "RandomForest (sklearn defaults: 100 trees, gini, no max_depth)", "seed": SEED, "train_time_s": round(train_time, 3)}
    preds = {}
    for name in ("train", "val", "test"):
        results[name], preds[name] = evaluate(model, X[name], y[name])

    t0 = time.perf_counter()
    model.predict(X["test"])
    results["test_inference_ms_per_sample"] = round(1000 * (time.perf_counter() - t0) / len(X["test"]), 4)

    report = classification_report(y["test"], preds["test"], labels=LABELS, digits=4, zero_division=0)

    # Reference only: random (shuffled) split, to show the leakage the day-based split avoids.
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(df[feats], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=SEED)
    rf_rand = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(Xr_tr, yr_tr)
    results["random_split_reference"], _ = evaluate(rf_rand, Xr_te, yr_te.to_numpy())

    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    results["feature_importance"] = imp.round(4).to_dict()

    (RES_DIR / "rf_baseline.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (RES_DIR / "rf_test_classification_report.txt").write_text(report, encoding="utf-8")

    print(f"Train time: {train_time:.2f}s")
    for name in ("train", "val", "test"):
        r = results[name]
        print(f"{name:5s} macro-F1={r['macro_f1']:.4f} acc={r['accuracy']:.4f} per-class={r['per_class_f1']}")
    rr = results["random_split_reference"]
    print(f"random-split reference: macro-F1={rr['macro_f1']:.4f} acc={rr['accuracy']:.4f}")
    print("\nTest classification report:\n" + report)
    print("Test confusion matrix:\n", confusion_matrix(y["test"], preds["test"], labels=LABELS))
    print("\nFeature importance:\n", imp.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, norm, title in ((axes[0], None, "counts"), (axes[1], "true", "row-normalized")):
        ConfusionMatrixDisplay.from_predictions(y["test"], preds["test"], labels=LABELS, normalize=norm, ax=ax, colorbar=False, cmap="Blues", values_format=".2f" if norm else "d")
        ax.set_title(f"RF test confusion matrix ({title})")
        ax.set_xlabel("Predicted occupants")
        ax.set_ylabel("True occupants")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rf_confusion_matrix.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    imp[::-1].plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Gini importance")
    ax.set_title("Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rf_feature_importance.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    names = ["Random split\n(leaky)", "Day split\nval", "Day split\ntest"]
    vals = [rr["macro_f1"], results["val"]["macro_f1"], results["test"]["macro_f1"]]
    ax.bar(names, vals, color=["#999999", "#55A868", "#4C72B0"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Random Forest: random vs. day-based split")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rf_split_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
