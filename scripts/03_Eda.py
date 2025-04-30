# 03_EDA.py – Exploratory Data Analysis with shared config paths
"""Exploratory Data Analysis & visualization for the engineered dataset.

Run from project root:
    python -m scripts.03_EDA
…או מתוך *scripts/* ישירות:
    python 03_EDA.py

Loads the engineered Excel produced by 002_Feature_Engineering.py and saves
plots to *plots/* (path provided by `config.PLOT_DIR`).
"""
import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure project root on path & import shared paths
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import OUTPUT_XLSX, PLOT_DIR

# ---------------------------------------------------------------------------
# Style & warnings
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="deep")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Helper – save figure
# ---------------------------------------------------------------------------

def save_fig(fig, name: str):
    """Save `fig` into PLOT_DIR as png."""
    outfile = PLOT_DIR / f"{name}.png"
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"✓ Saved {outfile.relative_to(project_root)}")

# ---------------------------------------------------------------------------
# Main EDA routine
# ---------------------------------------------------------------------------

def main():
    if not OUTPUT_XLSX.exists():
        raise FileNotFoundError(
            f"{OUTPUT_XLSX} not found. Run 002_Feature_Engineering.py first or set OUTPUT_XLSX correctly in config."
        )

    # Ensure plot directory exists
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data ----------------------------------------------------------
    df = pd.read_excel(OUTPUT_XLSX)

    # 2. Cancellation distribution -----------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="was_canceled", ax=ax, palette="Set2")
    ax.set_title("Cancellation count")
    ax.set_xlabel("Was canceled")
    ax.set_ylabel("Count")
    save_fig(fig, "cancel_count")

    # 3. Correlation heatmap (numeric) -------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Pearson correlation heatmap (numeric features)")
    save_fig(fig, "correlation_heatmap")

    # 4. Histograms & KDEs for key numeric vars ----------------------------
    key_numeric = ["distance_km", "num_diagnoses", "num_missing_labs"]
    for col in key_numeric:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=df, x=col, hue="was_canceled", kde=True, ax=ax, bins=30)
        ax.set_title(f"Distribution of {col}")
        save_fig(fig, f"hist_{col}")

    # 5. Boxplots for numeric vs. target -----------------------------------
    for col in key_numeric:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="was_canceled", y=col, ax=ax)
        ax.set_title(f"{col} by cancellation status")
        save_fig(fig, f"box_{col}")

    # 6. Categorical vs. cancellation rate ---------------------------------
    cat_cols = [
        "surgery_weekday",
        "season",
        "distance_bucket",
        "near_holiday",
    ]
    for col in cat_cols:
        rate = (
            df.groupby(col, observed=False)["was_canceled"].mean().sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(x=rate.index, y=rate.values, ax=ax)
        ax.set_title(f"Cancellation rate by {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Cancellation rate")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        save_fig(fig, f"rate_{col}")

    print("\nEDA complete – plots saved to", PLOT_DIR)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
