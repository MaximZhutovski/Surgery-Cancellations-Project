# config.py – resilient project-root detection and shared paths
"""
Shared path configuration for the Surgery_Cancellation project.

This file may sit **either** in the project root _or_ inside *scripts/*.
The logic below finds the real project root automatically so that the
paths to *data/*, *plots/*, and *results/* are always correct.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Detect project root
# ---------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent

# If the config file is inside *scripts*, ascend one level; else stay put.
if script_dir.name.lower() == "scripts":
    PROJECT_ROOT = script_dir.parent
else:
    PROJECT_ROOT = script_dir

# Allow override via environment variable
PROJECT_ROOT = Path(os.getenv("BASE_DIR", PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Standard directories
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
PLOT_DIR = Path(os.getenv("PLOT_DIR", PROJECT_ROOT / "plots"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results")) # Define results directory

# ---------------------------------------------------------------------------
# Key files
# ---------------------------------------------------------------------------
INPUT_XLSX  = Path(os.getenv("INPUT_XLSX",  DATA_DIR / "Surgery_Data.xlsx"))
OUTPUT_XLSX = Path(os.getenv("OUTPUT_XLSX", DATA_DIR / "surgery_data_engineered.xlsx"))

# ---------------------------------------------------------------------------
# Ensure folders exist
# ---------------------------------------------------------------------------
# Make sure all standard directories are created if they don't exist
for d in (DATA_DIR, PLOT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Configuration Paths ---")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR    : {DATA_DIR}")
    print(f"PLOT_DIR    : {PLOT_DIR}")
    print(f"RESULTS_DIR : {RESULTS_DIR}") # Print results dir
    print("-" * 20)
    print(f"INPUT_XLSX  : {INPUT_XLSX}")
    print(f"  Exists?   : {INPUT_XLSX.exists()}")
    print(f"OUTPUT_XLSX : {OUTPUT_XLSX}")
    print(f"  Exists?   : {OUTPUT_XLSX.exists()}")
    print("-" * 20)