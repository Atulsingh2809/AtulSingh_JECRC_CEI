"""Exploratory data analysis helpers — charts saved to reports/figures/."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TARGET_COL
from src.utils import replace_select_with_nan

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


def _save_fig(name: str, output_dir: Path | None = None) -> Path:
    out_dir = output_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)
    return path


def plot_class_balance(df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    """Bar chart of converted vs not converted."""
    counts = df[TARGET_COL].value_counts().sort_index()
    labels = ["Not Converted (0)", "Converted (1)"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_title("Class Balance — Lead Conversion")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{val:,}\n({100 * val / len(df):.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    return _save_fig("01_class_balance.png", output_dir)


def plot_conversion_by_category(
    df: pd.DataFrame,
    column: str,
    output_dir: Path | None = None,
    min_count: int = 30,
    top_n: int = 15,
    filename: str | None = None,
) -> Path:
    """Horizontal bar chart of conversion rate by categorical column."""
    stats = (
        df.groupby(column)[TARGET_COL]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "conversion_rate"})
    )
    stats = stats[stats["count"] >= min_count].sort_values("conversion_rate", ascending=True)
    if len(stats) > top_n:
        stats = stats.tail(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(stats))))
    ax.barh(stats[column].astype(str), stats["conversion_rate"], color="#55A868")
    ax.axvline(df[TARGET_COL].mean(), color="red", linestyle="--", label="Overall avg")
    ax.set_xlabel("Conversion Rate")
    ax.set_title(f"Conversion Rate by {column}")
    ax.legend()
    fname = filename or f"02_conversion_by_{column.lower().replace(' ', '_')[:40]}.png"
    return _save_fig(fname, output_dir)


def plot_numeric_distributions(
    df: pd.DataFrame,
    columns: list[str],
    output_dir: Path | None = None,
    filename: str = "03_numeric_distributions.png",
) -> Path:
    """Histograms of numeric features split by conversion status."""
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        if col not in df.columns:
            continue
        for label, subset in df.groupby(TARGET_COL):
            sns.histplot(
                subset[col].dropna(),
                ax=ax,
                label=f"Converted={label}",
                kde=True,
                stat="density",
                common_norm=False,
                alpha=0.5,
            )
        ax.set_title(col)
        ax.legend()

    return _save_fig(filename, output_dir)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    filename: str = "04_correlation_heatmap.png",
) -> Path:
    """Correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    if TARGET_COL in df.columns and TARGET_COL not in numeric.columns:
        numeric[TARGET_COL] = df[TARGET_COL]

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True)
    ax.set_title("Correlation Heatmap — Numeric Features")
    return _save_fig(filename, output_dir)


def plot_null_summary(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    filename: str = "05_null_value_summary.png",
) -> Path:
    """Bar chart of null + 'Select' percentage per column."""
    work = replace_select_with_nan(df)
    null_pct = work.isna().mean().sort_values(ascending=True)
    null_pct = null_pct[null_pct > 0]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(null_pct))))
    ax.barh(null_pct.index, null_pct.values * 100, color="#8172B3")
    ax.axvline(40, color="red", linestyle="--", label="40% threshold")
    ax.set_xlabel("Missing / Select (%)")
    ax.set_title("Null & 'Select' Value Summary")
    ax.legend()
    return _save_fig(filename, output_dir)


def run_full_eda(df: pd.DataFrame, output_dir: Path | None = None) -> list[Path]:
    """
    Generate all EDA charts and return list of saved figure paths.
    """
    out = output_dir or FIGURES_DIR
    paths = []

    paths.append(plot_class_balance(df, out))
    for col in [
        "Lead Origin",
        "Lead Source",
        "Last Activity",
        "Specialization",
        "City",
        "What is your current occupation",
    ]:
        if col in df.columns:
            paths.append(plot_conversion_by_category(df, col, out))

    numeric_cols = [
        "TotalVisits",
        "Total Time Spent on Website",
        "Page Views Per Visit",
    ]
    paths.append(plot_numeric_distributions(df, numeric_cols, out))
    paths.append(plot_correlation_heatmap(df, out))
    paths.append(plot_null_summary(df, out))

    logger.info("Generated %d EDA figures in %s", len(paths), out)
    return paths


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.utils import setup_logging

    setup_logging()
    data = load_raw_data(replace_select=False)
    saved = run_full_eda(data)
    print(f"Saved {len(saved)} figures to {FIGURES_DIR}")
