# 002_Feature_Engineering.py – full feature set with robust was_canceled logic
"""High-quality feature engineering for the surgery-cancellation dataset.

* File paths are loaded from config.py (no hardcoded paths).
* Corrected was_canceled logic: Comparison after converting to string+strip to prevent type mismatches.
* All original features are preserved (with English names).
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import holidays # Import the holidays library

# ---------------------------------------------------------------------------
# Ensure project root on PYTHONPATH & import shared paths
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import INPUT_XLSX, OUTPUT_XLSX

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Original HOLIDAYS_IL list removed/replaced below

DISTANCE_BINS = [0, 5, 10, 20, 50, 100, 200, 500]

# Hebrew → English columns map
COLS_MAP = {
    "ת\"ז מותממת": "patient_id",
    "תאריך פתיחת בקשה": "request_date",
    "תאריך ביצוע ניתוח": "surgery_date",
    "מספר ניתוח תכנון": "plan_id",
    "חדר": "room",
    "הרדמה": "anesthesia",
    "קוד ניתוח": "procedure_code",
    "גיל המטופל בזמן הניתוח": "age",
    "BMI": "bmi",
    "תרופות קבועות": "medications",
    "אבחנות רקע": "diagnoses",
    "ALBUMIN": "albumin",
    "PT-INR": "pt_inr",
    "POTASSIUM": "potassium",
    "SODIUM": "sodium",
    "HB": "hb",
    "WBC": "wbc",
    "PLT": "plt",
    "עיר": "city",
    "אתר ניתוח": "surgery_site",
    "distance_km": "distance_km",
}

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Define the relevant year range for your data ---
DATA_START_YEAR = 2020
DATA_END_YEAR = 2024

# --- Create a dynamic list of holidays for the relevant years ---
try:
    # Create an Israeli holidays object for the year range
    il_holidays_generator = holidays.IL(years=range(DATA_START_YEAR, DATA_END_YEAR + 1))
    # Convert to a Set of datetime dates for quick lookup
    HOLIDAYS_IL_DYNAMIC = set(pd.to_datetime(list(il_holidays_generator.keys())))
    print(f"Generated {len(HOLIDAYS_IL_DYNAMIC)} Israeli holiday dates for years {DATA_START_YEAR}-{DATA_END_YEAR}")
except Exception as e:
    print(f"Error creating dynamic holiday list: {e}")
    print("Proceeding with an empty list.")
    HOLIDAYS_IL_DYNAMIC = set()

# ---------------------------------------------------------------------------
# Helper Function
# ---------------------------------------------------------------------------

def add_cancel_rate(df: pd.DataFrame, cat: str, rate_name: str) -> None:
    """Add mean cancellation rate for category *cat*."""
    # Ensure the category column exists before using it
    if cat not in df.columns:
        print(f"Warning: Category column '{cat}' not found for calculating rate '{rate_name}'. Skipping.")
        df[rate_name] = np.nan # Or 0, depending on how you want to handle it
        return
    # Handle potential missing values in the category to avoid errors in groupby
    # Calculate only on non-missing values (observed=False ensures category is considered if present)
    df[rate_name] = df.groupby(cat, observed=False)["was_canceled"].transform("mean")

# ---------------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Input Excel file not found: {INPUT_XLSX}")

    # --- Load Data -----------------------------------------------------------
    print("Loading data...")
    xls = pd.ExcelFile(INPUT_XLSX)
    df_all = pd.read_excel(xls, sheet_name=0).rename(columns=COLS_MAP)
    df_cancel = pd.read_excel(xls, sheet_name=1)
    print(f" Loaded {len(df_all):,} records from the main sheet.")
    print(f" Loaded {len(df_cancel):,} records from the cancellation sheet.")


    # --- Robust was_canceled Calculation -------------------------------------
    print("Calculating 'was_canceled'...")
    # Ensure the key column exists in both DataFrames
    if 'plan_id' not in df_all.columns:
         raise KeyError("Column 'plan_id' (expected result of rename) not found in df_all.")
    # Use the original Hebrew name for the cancellation sheet before potential rename
    original_cancel_id_col = "מספר ניתוח תכנון"
    if original_cancel_id_col not in df_cancel.columns:
         raise KeyError(f"Column '{original_cancel_id_col}' not found in df_cancel.")

    # Use a set for slightly better performance with isin
    plans_cancel_set = set(df_cancel[original_cancel_id_col].astype(str).str.strip())
    plans_all_series = df_all["plan_id"].astype(str).str.strip()
    df_all["was_canceled"] = plans_all_series.isin(plans_cancel_set)
    print(f" Marked {df_all['was_canceled'].sum():,} records as canceled.")

    # --- Dates & Calendar Features -------------------------------------------
    print("Processing dates and calendar features...")
    df_all["request_date"] = pd.to_datetime(df_all["request_date"], errors="coerce")
    df_all["surgery_date"] = pd.to_datetime(df_all["surgery_date"], errors="coerce")

    # Handle potential missing dates before extracting features
    valid_dates_mask = df_all["surgery_date"].notna()
    df_all["surgery_weekday"] = np.nan
    df_all.loc[valid_dates_mask, "surgery_weekday"] = df_all.loc[valid_dates_mask, "surgery_date"].dt.day_name()

    df_all["is_weekend"] = df_all["surgery_weekday"].isin(["Friday", "Saturday"]) # Handles NaN as False

    month_to_season = {1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Spring", 6:"Summer",
                       7:"Summer", 8:"Summer", 9:"Fall", 10:"Fall", 11:"Fall", 12:"Winter"}
    df_all["season"] = np.nan
    df_all.loc[valid_dates_mask, "season"] = df_all.loc[valid_dates_mask, "surgery_date"].dt.month.map(month_to_season)


    # --- Demographics & Lab Features -----------------------------------------
    print("Creating demographic and lab features...")
    lab_cols = ["albumin", "pt_inr", "potassium", "sodium", "hb", "wbc", "plt"]
    # Ensure all lab columns exist
    existing_lab_cols = [col for col in lab_cols if col in df_all.columns]
    if len(existing_lab_cols) != len(lab_cols):
        print(f"Warning: Missing lab columns: {list(set(lab_cols) - set(existing_lab_cols))}")

    df_all["age_decade"] = pd.NA # Start with missing value
    if 'age' in df_all.columns and df_all['age'].notna().any():
         # Temporarily fill NA for calculation, then reset NA if original was NA
         df_all["age_decade"] = (df_all["age"].fillna(-1) // 10).astype("Int64") * 10
         df_all.loc[df_all["age"].isna(), "age_decade"] = pd.NA

    # If 'bmi' column doesn't exist, assume it's missing
    df_all["bmi_missing"] = df_all["bmi"].isna() if 'bmi' in df_all.columns else True
    df_all["num_missing_labs"] = df_all[existing_lab_cols].isna().sum(axis=1)
    df_all["has_missing_labs"] = df_all["num_missing_labs"] > 0

    # Function to handle counting comma-separated items
    def count_items(text_series):
        # Handles NaN and empty strings
        return text_series.fillna('').astype(str).apply(lambda x: len(x.split(',')) if x.strip() else 0)

    if 'medications' in df_all.columns:
        df_all["num_medications"] = count_items(df_all["medications"])
    else:
        df_all["num_medications"] = 0
        print("Warning: 'medications' column not found, setting num_medications to 0.")

    if 'diagnoses' in df_all.columns:
        df_all["num_diagnoses"] = count_items(df_all["diagnoses"])
    else:
         df_all["num_diagnoses"] = 0
         print("Warning: 'diagnoses' column not found, setting num_diagnoses to 0.")


    # --- Historical Cancellation Rates ---------------------------------------
    print("Calculating historical cancellation rates...")
    # Handle cases where room/site columns might be missing
    if 'surgery_site' in df_all.columns and 'room' in df_all.columns:
        df_all["site_room"] = df_all["surgery_site"].astype(str) + "_" + df_all["room"].astype(str)
        add_cancel_rate(df_all, "site_room", "site_room_cancel_rate")
    else:
        print("Warning: 'surgery_site' or 'room' column missing. Cannot calculate site_room_cancel_rate.")
        df_all["site_room_cancel_rate"] = np.nan

    add_cancel_rate(df_all, "anesthesia", "anesthesia_cancel_rate")
    add_cancel_rate(df_all, "procedure_code", "procedure_code_cancel_rate")
    # Weekday cancellation rate - ensure the column exists
    if "surgery_weekday" in df_all.columns:
        add_cancel_rate(df_all, "surgery_weekday", "weekday_cancel_rate")
    else:
        df_all["weekday_cancel_rate"] = np.nan


    # --- Holiday Proximity (Using the dynamic list) ---
    print("Calculating holiday proximity...")
    # Ensure surgery_date is datetime again (belt and suspenders)
    df_all["surgery_date"] = pd.to_datetime(df_all["surgery_date"], errors='coerce')

    # Function to check holiday proximity (compares date part only)
    def check_near_holiday(surg_date, holidays_set):
        if pd.isna(surg_date) or not holidays_set:
            return False
        surg_dt_date_only = surg_date.date() # Convert to date object
        # Compare date part only
        return any(abs((surg_dt_date_only - hol_date.date()).days) <= 3 for hol_date in holidays_set)

    # Apply the function to the column
    df_all["near_holiday"] = df_all["surgery_date"].apply(check_near_holiday, holidays_set=HOLIDAYS_IL_DYNAMIC)


    # --- Distance Features ----------------------------------------------------
    print("Creating distance-based features...")
    if 'distance_km' in df_all.columns:
        # Removed the 'cancel_per_km' feature due to potential data leakage

        # Ensure the column is numeric and convert to float if necessary
        df_all['distance_km'] = pd.to_numeric(df_all['distance_km'], errors='coerce')

        labels_km = [f"{DISTANCE_BINS[i]}–{DISTANCE_BINS[i+1]} km" for i in range(len(DISTANCE_BINS)-1)]
        # include_lowest=True is important to catch 0 distance
        # right=False makes bins [low, high) - includes lower bound
        df_all["distance_bucket"] = pd.cut(df_all["distance_km"], bins=DISTANCE_BINS, labels=labels_km, right=False, include_lowest=True)
        add_cancel_rate(df_all, "distance_bucket", "distance_bucket_cancel_rate")
    else:
        print("Warning: 'distance_km' column not found. Skipping distance features.")
        df_all["distance_bucket"] = pd.NA
        df_all["distance_bucket_cancel_rate"] = np.nan


    # --- Preview Generated Features ------------------------------------------
    print("\nPreview of generated features (first 5 rows):")
    # Update preview list - removed cancel_per_km and reflect checks
    preview_cols = [
        "plan_id", # For context
        "was_canceled", "surgery_date", "surgery_weekday", "is_weekend", "season",
        "age_decade", "bmi_missing",
        "num_missing_labs", "has_missing_labs", "num_medications", "num_diagnoses",
        "site_room_cancel_rate", "anesthesia_cancel_rate",
        "procedure_code_cancel_rate", "weekday_cancel_rate",
        "near_holiday",
        "distance_km", "distance_bucket", "distance_bucket_cancel_rate",
    ]
    # Show only columns that actually exist in the final DataFrame
    existing_preview_cols = [col for col in preview_cols if col in df_all.columns]
    print(df_all[existing_preview_cols].head().to_markdown(index=False, numalign="left", stralign="left"))


    # --- Save Engineered Data ------------------------------------------------
    choice = input("\n💬 Save the engineered file to Excel? (y/n): ").strip().lower()
    if choice == 'y':
        try:
            OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
            # Attempt to convert object columns that might cause saving issues
            # Prefer specific conversion if problems arise, but this is general
            for col in df_all.select_dtypes(include=['object']).columns:
                 # Avoid converting datetimes already handled
                 if pd.api.types.is_datetime64_any_dtype(df_all[col]):
                      continue
                 try:
                      # Try converting to string, might resolve saving issues
                      df_all[col] = df_all[col].astype(str)
                 except Exception as e_conv:
                      print(f"  Warning: Could not convert column '{col}' to string for saving: {e_conv}")

            print(f"\nSaving file to: {OUTPUT_XLSX}...")
            # Use openpyxl engine, recommended for .xlsx files
            df_all.to_excel(OUTPUT_XLSX, sheet_name="features", index=False, engine='openpyxl')
            print(f"\n✅ Successfully saved → {OUTPUT_XLSX.relative_to(project_root)}")
        except Exception as e:
            print(f"\n❌ Error saving file: {e}")
            print("There might be a problematic data type in one of the columns.")
    else:
        print("\n❎ File was not saved.")

# --- Run Main Function -----------------------------------------------------
if __name__ == "__main__":
    main()