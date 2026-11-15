"""Dataset analysis: structure, class balance, per-day/per-split distributions, sensor plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import FIG_DIR, LABELS, RES_DIR, TARGET, feature_columns, get_split, load_data


def main():
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    df = load_data()
    feats = feature_columns(df)
    lines = []

    lines.append(f"Samples: {len(df)}, features: {len(feats)}, missing values: {int(df.isna().sum().sum())}")
    lines.append(f"Features: {feats}")
    lines.append(f"Time range: {df.Timestamp.min()} -> {df.Timestamp.max()}")
    lines.append(f"Median sampling interval: {df.Timestamp.diff().median()}")

    counts = df[TARGET].value_counts().reindex(LABELS)
    lines.append("\nClass distribution (overall):")
    lines.append(pd.DataFrame({"count": counts, "ratio": (counts / len(df)).round(4)}).to_string())

    lines.append("\nSamples per day and class:")
    lines.append(pd.crosstab(df.Date, df[TARGET], margins=True).to_string())

    split = get_split(df)
    lines.append("\nSamples per split and class:")
    tab = pd.DataFrame({k: v[TARGET].value_counts().reindex(LABELS, fill_value=0) for k, v in split.items()}).T
    tab["total"] = tab.sum(axis=1)
    tab["share"] = (tab["total"] / len(df)).round(3)
    lines.append(tab.to_string())

    lines.append("\nFeature summary:")
    lines.append(df[feats].describe().T.round(3).to_string())

    lines.append("\nMean feature value per class:")
    lines.append(df.groupby(TARGET)[feats].mean().T.round(3).to_string())

    text = "\n".join(lines)
    (RES_DIR / "eda_summary.txt").write_text(text, encoding="utf-8")
    print(text)

    # Class distribution bar chart
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar([str(c) for c in LABELS], counts.values, color="#4C72B0")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / len(df):.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Number of occupants")
    ax.set_ylabel("Samples")
    ax.set_title("Class distribution")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)

    # Per-split class distribution
    fig, ax = plt.subplots(figsize=(6, 3.5))
    tab[LABELS].plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
    ax.set_ylabel("Samples")
    ax.set_title("Class distribution per split (day-based)")
    ax.legend(title="Occupants")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "split_distribution.png", dpi=200)
    plt.close(fig)

    # Correlation matrix
    corr = df[feats + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Feature correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation.png", dpi=200)
    plt.close(fig)

    # Sensor time series for one occupied day
    day = df[df.Date == "2017/12/22"]
    cols = [("S1_Light", "Light (lux)"), ("S1_Sound", "Sound (V)"), ("S5_CO2", "CO2 (ppm)"), ("S1_Temp", "Temp (C)")]
    fig, axes = plt.subplots(len(cols) + 1, 1, figsize=(9, 8), sharex=True)
    for ax, (c, lab) in zip(axes, cols):
        ax.plot(day.Timestamp, day[c], lw=0.8)
        ax.set_ylabel(lab, fontsize=8)
    axes[-1].step(day.Timestamp, day[TARGET], where="post", color="k")
    axes[-1].set_ylabel("Occupants", fontsize=8)
    fig.suptitle("Sensor readings vs. occupancy on 2017-12-22")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "timeseries_2017-12-22.png", dpi=200)
    plt.close(fig)

    # Feature distributions per class
    show = ["S1_Light", "S2_Light", "S1_Sound", "S5_CO2", "S5_CO2_Slope", "S1_Temp"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, c in zip(axes.ravel(), show):
        ax.boxplot([df.loc[df[TARGET] == k, c] for k in LABELS], tick_labels=[str(k) for k in LABELS], showfliers=False)
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("Occupants", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_by_class.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
