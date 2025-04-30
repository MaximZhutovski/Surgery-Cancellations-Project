# 01_Data_Loading_and_Exploration.py
"""Quick data overview & missing‑value inspection.
Run from the project root:
    python -m scripts.01_Data_Loading_and_Exploration
or from *scripts/* directly:
    python 01_Data_Loading_and_Exploration.py
"""
import sys
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH so that `config` is found
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import INPUT_XLSX

print(f"Using input file: {INPUT_XLSX}")
if not INPUT_XLSX.exists():
    raise FileNotFoundError(f"Input Excel file not found: {INPUT_XLSX}")

# === Step 1: Load Excel data ===
xls = pd.ExcelFile(INPUT_XLSX)
df_all = pd.read_excel(xls, sheet_name=0)
df_canceled = pd.read_excel(xls, sheet_name=1)

# === Step 2: Basic Overview ===
print("\n=== Data Overview ===")
print(f"Total records (All Surgeries): {len(df_all):,}")
print(f"Total records (Canceled Surgeries): {len(df_canceled):,}")
print(f"Unique patients (All): {df_all['ת\"ז מותממת'].nunique():,}")
print(f"Unique patients who canceled: {df_canceled['ת\"ז מותממת'].nunique():,}")

# === Step 3: Missing Values Analysis for All Surgeries ===
missing_all = (df_all.isnull().mean() * 100).round(1)
significant_missing_all = missing_all[missing_all > 20].sort_values(ascending=False)
print("\n=== Significant Missing Values (>20%) - All Surgeries ===")
print(significant_missing_all.astype(str) + "%")

# === Step 4: Missing Values Analysis for Canceled Surgeries ===
missing_canceled = (df_canceled.isnull().mean() * 100).round(1)
significant_missing_canceled = missing_canceled[missing_canceled > 20].sort_values(ascending=False)
print("\n=== Significant Missing Values (>20%) - Canceled Surgeries ===")
print(significant_missing_canceled.astype(str) + "%")

# === Step 5: Cancellation rate (by patient ID) ===
df_all['was_canceled'] = df_all['ת"ז מותממת'].isin(df_canceled['ת"ז מותממת'])
overall_cancel_rate = df_all['was_canceled'].mean() * 100
print(f"\nOverall Cancellation Rate: {overall_cancel_rate:.2f}%")

# === Step 6: Cancellation Analysis by Age Group ===
df_all['age_group'] = pd.cut(
    df_all['גיל המטופל בזמן הניתוח'],
    bins=[0, 18, 35, 50, 65, 80, 120],
    labels=['Child', 'Young Adult', 'Adult', 'Mid Age', 'Senior', 'Elderly']
)
age_cancel_rate = (df_all.groupby('age_group')['was_canceled'].mean() * 100).round(1)
print("\n=== Cancellation Rate by Age Group ===")
print(age_cancel_rate.astype(str) + "%")

# === Step 7: Cancellation Analysis by Weekday ===
df_all['weekday'] = pd.to_datetime(df_all['תאריך ביצוע ניתוח']).dt.day_name()
weekday_cancel_rate = (df_all.groupby('weekday')['was_canceled'].mean() * 100).round(1)
print("\n=== Cancellation Rate by Day of Week ===")
print(weekday_cancel_rate.astype(str) + "%")

# === Step 8: Timing Statistics (Days from Request to Surgery) ===
timing_stats = df_all['ימים מתאריך פתיחת בקשה ועד תאריך ביצוע ניתוח'].describe().round(1)
print("\n=== Days from Request to Surgery (Statistics) ===")
print(timing_stats[['mean', 'std', 'min', 'max']])
