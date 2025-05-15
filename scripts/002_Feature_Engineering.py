# 002_Feature_Engineering.py – extended feature set
"""High-quality feature engineering for the surgery-cancellation dataset.
* File paths are loaded from config.py.
* Corrected was_canceled logic.
* All original features are preserved (with English names where mapped).
* Added 'wait_days' calculation.
* NEW: Socioeconomic cluster (requires external LMS file).
* NEW: Days to/from nearest holiday.
* NEW: Is surgery after weekend/holiday flag.
* NEW: Smoothed historical cancellation rates.
* NEW: Categorization for 'wait_days' and 'bmi'.
* NEW: Cleaned 'site_room'.
* NEW: Changed 'num_missing_labs' to 'num_existing_labs'.
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import holidays
import re # For regex cleaning of site_room
# import logging # Removed for now, using print

# ---------------------------------------------------------------------------
# Setup: Add project root to PYTHONPATH and import config
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import (RAW_DATA_XLSX, ENGINEERED_DATA_XLSX, DATA_DIR, COLS_MAP,
                        LMS_SOCIOECONOMIC_XLSX, LMS_SHEET_NAME,
                        LMS_CITY_COL, LMS_CLUSTER_COL)
    print("Successfully imported paths from config.py")
except ImportError:
    print("CRITICAL: Could not import 'config'. Ensure config.py is in the 'scripts' directory and accessible.")
    print("Or, one of the expected variables (RAW_DATA_XLSX, etc.) is missing from config.py.")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL: An unexpected error occurred during config import: {e}")
    sys.exit(1)

print(f"--- Feature Engineering (Script 002) ---")
print(f"Using raw data input: {RAW_DATA_XLSX}")
print(f"Engineered data output will be: {ENGINEERED_DATA_XLSX}")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DISTANCE_BINS = [0, 5, 10, 20, 50, 100, 200, 500, np.inf]
WAIT_DAYS_BINS = [-np.inf, 7, 30, 90, 365, np.inf]
WAIT_DAYS_LABELS = ["0-7 days", "8-30 days", "31-90 days", "91-365 days", "365+ days"]
BMI_BINS = [0, 18.5, 24.9, 29.9, 34.9, 39.9, np.inf]
BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese_I", "Obese_II", "Obese_III"]
DEFAULT_SMOOTHING_ALPHA = 1

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_START_YEAR = 2018
DATA_END_YEAR = 2025
try:
    il_holidays_generator = holidays.IL(years=range(DATA_START_YEAR, DATA_END_YEAR + 1))
    HOLIDAYS_IL_DYNAMIC = set(pd.to_datetime(list(il_holidays_generator.keys())).date)
    print(f"Generated {len(HOLIDAYS_IL_DYNAMIC)} Israeli holiday dates for years {DATA_START_YEAR}-{DATA_END_YEAR}")
except Exception as e:
    print(f"Error creating dynamic holiday list: {e}. Proceeding with an empty list.")
    HOLIDAYS_IL_DYNAMIC = set()

# ---------------------------------------------------------------------------
# Helper Function for Smoothed Cancellation Rates
# ---------------------------------------------------------------------------
def add_smoothed_cancel_rate(original_df: pd.DataFrame, cat_col: str, rate_name: str, alpha: int = DEFAULT_SMOOTHING_ALPHA) -> None:
    if cat_col not in original_df.columns:
        print(f"Warning: Category column '{cat_col}' not found for rate '{rate_name}'. Skipping.")
        original_df[rate_name] = np.nan
        return
    if 'was_canceled' not in original_df.columns:
        print(f"Warning: 'was_canceled' column not found. Cannot calculate rate '{rate_name}'. Skipping.")
        original_df[rate_name] = np.nan
        return

    df_copy = original_df[[cat_col, 'was_canceled']].copy()
    df_copy[cat_col] = df_copy[cat_col].astype(str)
    df_copy.replace('nan', np.nan, inplace=True)
    df_copy.dropna(subset=[cat_col, 'was_canceled'], inplace=True)

    if df_copy.empty:
        print(f"Warning: No valid data for '{cat_col}' after NA drop to calculate '{rate_name}'. Filling with neutral 0.5.")
        original_df[rate_name] = 0.5
        return

    agg_stats = df_copy.groupby(cat_col, observed=False)['was_canceled'].agg(
        canceled_sum='sum',
        total_count='count'
    )
    K = 2
    smoothed_rates = (agg_stats['canceled_sum'] + alpha) / (agg_stats['total_count'] + K * alpha)
    
    original_df[rate_name] = original_df[cat_col].astype(str).map(smoothed_rates)
    original_df[rate_name] = pd.to_numeric(original_df[rate_name], errors='coerce')

    overall_canceled_sum_for_fill = df_copy['was_canceled'].sum()
    overall_total_count_for_fill = len(df_copy)
    global_smoothed_rate_for_fill = (overall_canceled_sum_for_fill + alpha) / (overall_total_count_for_fill + K * alpha) if (overall_total_count_for_fill + K * alpha) != 0 else 0.5
    
    original_df[rate_name].fillna(global_smoothed_rate_for_fill, inplace=True)
    print(f"Calculated '{rate_name}' for '{cat_col}'. Filled NaNs with global smoothed rate: {global_smoothed_rate_for_fill:.4f}")

# ---------------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------------
def main() -> None:
    if not RAW_DATA_XLSX.exists():
        raise FileNotFoundError(f"Input Excel file not found: {RAW_DATA_XLSX}")

    print("Loading data...")
    xls = pd.ExcelFile(RAW_DATA_XLSX)
    df_all = pd.read_excel(xls, sheet_name=0).rename(columns=COLS_MAP) # Uses COLS_MAP from config
    df_cancel = pd.read_excel(xls, sheet_name=1)
    print(f" Loaded {len(df_all):,} records from the main sheet.")
    print(f" Loaded {len(df_cancel):,} records from the cancellation sheet.")

    print("Calculating 'was_canceled'...")
    if 'plan_id' not in df_all.columns:
         raise KeyError("'plan_id' not in df_all.")
    original_cancel_id_col = "מספר ניתוח תכנון"
    if original_cancel_id_col not in df_cancel.columns:
         raise KeyError(f"'{original_cancel_id_col}' not found in df_cancel.")
    plans_cancel_set = set(df_cancel[original_cancel_id_col].astype(str).str.strip())
    df_all["was_canceled"] = df_all["plan_id"].astype(str).str.strip().isin(plans_cancel_set)
    print(f" Marked {df_all['was_canceled'].sum():,} records as canceled.")

    print("Processing dates and calendar features...")
    df_all["request_date"] = pd.to_datetime(df_all["request_date"], errors="coerce")
    df_all["surgery_date"] = pd.to_datetime(df_all["surgery_date"], errors="coerce")

    if 'request_date' in df_all.columns and 'surgery_date' in df_all.columns:
        df_all["wait_days"] = (df_all["surgery_date"] - df_all["request_date"]).dt.days
        df_all["wait_days_category"] = pd.cut(df_all["wait_days"], bins=WAIT_DAYS_BINS, labels=WAIT_DAYS_LABELS, right=False).astype(object)
        df_all["wait_days_category"].fillna("__MISSING__", inplace=True)
        
        # Corrected print statement for mean_wait_days
        mean_wait_days_val = df_all['wait_days'].mean()
        mean_wait_days_str = f"{mean_wait_days_val:.2f}" if pd.notna(mean_wait_days_val) else "N/A"
        print(f" Calculated 'wait_days' and 'wait_days_category'. Min: {df_all['wait_days'].min()}, Max: {df_all['wait_days'].max()}, Mean: {mean_wait_days_str}")
    else:
        print("Warning: 'request_date' or 'surgery_date' not found. Cannot calculate 'wait_days'.")
        df_all["wait_days"] = np.nan
        df_all["wait_days_category"] = "__MISSING__" # Ensure it's a string placeholder

    valid_dates_mask = df_all["surgery_date"].notna()
    df_all["surgery_weekday"] = pd.NA # Start with object/NA to allow string names
    df_all.loc[valid_dates_mask, "surgery_weekday"] = df_all.loc[valid_dates_mask, "surgery_date"].dt.day_name()
    df_all["is_weekend"] = df_all["surgery_weekday"].isin(["Friday", "Saturday"])

    month_to_season = {1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Spring", 6:"Summer",
                       7:"Summer", 8:"Summer", 9:"Fall", 10:"Fall", 11:"Fall", 12:"Winter"}
    df_all["season"] = pd.NA # Start with object/NA
    df_all.loc[valid_dates_mask, "season"] = df_all.loc[valid_dates_mask, "surgery_date"].dt.month.map(month_to_season)
    
    print("Calculating holiday proximity features...")
    df_all["days_to_next_holiday"] = np.nan
    df_all["days_from_prev_holiday"] = np.nan
    df_all["is_surgery_after_holiday_weekend"] = False
    if HOLIDAYS_IL_DYNAMIC:
        sorted_holidays = sorted(list(HOLIDAYS_IL_DYNAMIC))
        for index, row in df_all.iterrows(): # Consider vectorizing if slow
            if pd.notna(row["surgery_date"]):
                surg_date_date_only = row["surgery_date"].date()
                next_hols = [h for h in sorted_holidays if h >= surg_date_date_only]
                if next_hols: df_all.loc[index, "days_to_next_holiday"] = (min(next_hols) - surg_date_date_only).days
                prev_hols = [h for h in sorted_holidays if h <= surg_date_date_only]
                if prev_hols: df_all.loc[index, "days_from_prev_holiday"] = (surg_date_date_only - max(prev_hols)).days
                day_before = surg_date_date_only - pd.Timedelta(days=1)
                if day_before in HOLIDAYS_IL_DYNAMIC or day_before.weekday() in [4, 5]: # Friday=4, Saturday=5
                    df_all.loc[index, "is_surgery_after_holiday_weekend"] = True
    df_all["days_to_next_holiday"].fillna(999, inplace=True)
    df_all["days_from_prev_holiday"].fillna(999, inplace=True)
    df_all["near_holiday"] = False
    if HOLIDAYS_IL_DYNAMIC:
        for index, row in df_all.iterrows(): # Consider vectorizing if slow
            if pd.notna(row["surgery_date"]):
                s_date = row["surgery_date"].date()
                df_all.loc[index, "near_holiday"] = any(abs((s_date - hol_date).days) <= 3 for hol_date in HOLIDAYS_IL_DYNAMIC)

    print("Creating demographic and lab features...")
    lab_cols = ["albumin", "pt_inr", "potassium", "sodium", "hb", "wbc", "plt"]
    existing_lab_cols = [col for col in lab_cols if col in df_all.columns]
    df_all["age_decade"] = pd.NA
    if 'age' in df_all.columns and df_all['age'].notna().any():
         df_all["age"] = pd.to_numeric(df_all["age"], errors='coerce')
         df_all.loc[df_all['age'].notna(), "age_decade"] = (df_all.loc[df_all['age'].notna(), "age"] // 10).astype("Int64") * 10
    df_all["bmi_missing"] = df_all["bmi"].isna() if 'bmi' in df_all.columns else True
    if 'bmi' in df_all.columns:
        df_all["bmi_category"] = pd.cut(pd.to_numeric(df_all["bmi"], errors='coerce'), bins=BMI_BINS, labels=BMI_LABELS, right=False).astype(object)
        df_all["bmi_category"].fillna("__MISSING__", inplace=True)
    else:
        df_all["bmi_category"] = "__MISSING__"
    if existing_lab_cols:
        df_all["num_existing_labs"] = df_all[existing_lab_cols].notna().sum(axis=1)
    else:
        df_all["num_existing_labs"] = 0
    df_all["has_missing_labs"] = (df_all["num_existing_labs"] < len(existing_lab_cols)) if existing_lab_cols else True
    def count_items(text_series):
        return text_series.fillna('').astype(str).apply(lambda x: len(x.split(',')) if x.strip() else 0)
    df_all["num_medications"] = count_items(df_all["medications"]) if 'medications' in df_all.columns else 0
    df_all["num_diagnoses"] = count_items(df_all["diagnoses"]) if 'diagnoses' in df_all.columns else 0

    print("Attempting to add socioeconomic feature...")
    if LMS_SOCIOECONOMIC_XLSX.exists() and 'city' in df_all.columns:
        try:
            df_lms = pd.read_excel(LMS_SOCIOECONOMIC_XLSX, sheet_name=LMS_SHEET_NAME)
            df_lms[LMS_CITY_COL_CLEANED] = df_lms[LMS_CITY_COL].str.strip() # Use a new temp col name
            df_all['city_clean_for_merge'] = df_all['city'].astype(str).str.strip()
            df_all = pd.merge(df_all, df_lms[[LMS_CITY_COL_CLEANED, LMS_CLUSTER_COL]],
                              left_on='city_clean_for_merge', right_on=LMS_CITY_COL_CLEANED, how='left') # Use cleaned LMS col
            df_all.rename(columns={LMS_CLUSTER_COL: 'socioeconomic_cluster'}, inplace=True)
            df_all.drop(columns=['city_clean_for_merge', LMS_CITY_COL_CLEANED], errors='ignore', inplace=True) # Drop temp cols
            if 'socioeconomic_cluster' in df_all.columns:
                df_all['socioeconomic_cluster'] = pd.to_numeric(df_all['socioeconomic_cluster'], errors='coerce')
                if df_all['socioeconomic_cluster'].isnull().any():
                    median_cluster = df_all['socioeconomic_cluster'].median()
                    df_all['socioeconomic_cluster'].fillna(median_cluster, inplace=True)
                    print(f"Added 'socioeconomic_cluster'. Filled NaNs with median: {median_cluster:.2f if pd.notna(median_cluster) else 'N/A'}")
                else: print("Added 'socioeconomic_cluster'. No NaNs to fill or all were non-numeric initially.")
            else:
                print("Warning: 'socioeconomic_cluster' column not created/retained after merge."); df_all['socioeconomic_cluster'] = np.nan
        except KeyError as ke:
             print(f"KeyError during socioeconomic feature creation (check LMS column names): {ke}. Column will be NaN.")
             df_all['socioeconomic_cluster'] = np.nan
        except Exception as e:
            print(f"Error adding socioeconomic feature: {e}. Column will be NaN."); df_all['socioeconomic_cluster'] = np.nan
    else:
        print(f"LMS file ({LMS_SOCIOECONOMIC_XLSX}) not found or 'city' column missing. Skipping socioeconomic feature.")
        df_all['socioeconomic_cluster'] = np.nan

    print("Calculating smoothed historical cancellation rates...")
    if 'surgery_site' in df_all.columns and 'room' in df_all.columns:
        df_all["site_room"] = df_all["surgery_site"].astype(str) + "_" + df_all["room"].astype(str)
        df_all['site_room'] = df_all['site_room'].str.replace(r'\s+', ' ', regex=True).str.strip()
        df_all['site_room'] = df_all['site_room'].str.replace(r'\s*(\d+)$', r'_\1', regex=True)
        print("Cleaned 'site_room' feature.")
        add_smoothed_cancel_rate(df_all, "site_room", "site_room_cancel_rate_smoothed")
    else:
        print("Warning: 'surgery_site' or 'room' missing. Cannot create/clean 'site_room' or its rate.")
        df_all["site_room"] = pd.NA; df_all["site_room_cancel_rate_smoothed"] = np.nan
    
    for cat, rate_name_suffix in [
        ("anesthesia", "anesthesia_cancel_rate_smoothed"),
        ("procedure_code", "procedure_code_cancel_rate_smoothed"),
        ("surgery_weekday", "weekday_cancel_rate_smoothed")]:
        if cat in df_all.columns: add_smoothed_cancel_rate(df_all, cat, rate_name_suffix)
        else: df_all[rate_name_suffix] = np.nan
            
    print("Creating distance-based features...")
    if 'distance_km' in df_all.columns:
        df_all['distance_km'] = pd.to_numeric(df_all['distance_km'], errors='coerce')
        labels_km = [f"{DISTANCE_BINS[i]}-{DISTANCE_BINS[i+1]} km" for i in range(len(DISTANCE_BINS)-2)] + [f"{DISTANCE_BINS[-2]}+ km"]
        df_all["distance_bucket"] = pd.cut(df_all["distance_km"], bins=DISTANCE_BINS, labels=labels_km, right=False, include_lowest=True).astype(object)
        df_all["distance_bucket"].fillna("__MISSING__", inplace=True)
        add_smoothed_cancel_rate(df_all, "distance_bucket", "distance_bucket_cancel_rate_smoothed")
    else:
        df_all["distance_bucket"] = "__MISSING__"; df_all["distance_bucket_cancel_rate_smoothed"] = np.nan

    print("\nPreview of some generated features (first 5 rows):")
    preview_cols = [
        "plan_id", "was_canceled", "site_room", "num_existing_labs", "socioeconomic_cluster",
        "wait_days_category", "bmi_category", "distance_bucket", "procedure_code_cancel_rate_smoothed"
    ]
    existing_preview_cols = [col for col in preview_cols if col in df_all.columns]
    if existing_preview_cols:
        print(df_all[existing_preview_cols].head().to_markdown(index=False, numalign="left", stralign="left"))

    choice = input("\n💬 Save the engineered file to Excel? (y/n): ").strip().lower()
    if choice == 'y':
        try:
            ENGINEERED_DATA_XLSX.parent.mkdir(parents=True, exist_ok=True)
            for col in df_all.select_dtypes(include=['object', 'category']).columns:
                if pd.api.types.is_categorical_dtype(df_all[col]):
                    if not all(isinstance(item, str) for item in df_all[col].cat.categories):
                        df_all[col] = df_all[col].astype(str)
            print(f"\nSaving file to: {ENGINEERED_DATA_XLSX}...")
            df_all.to_excel(ENGINEERED_DATA_XLSX, sheet_name="features_v3", index=False, engine='openpyxl')
            print(f"\n✅ Successfully saved → {ENGINEERED_DATA_XLSX.relative_to(project_root)}")
        except Exception as e:
            print(f"\n❌ Error saving file: {e}")
    else:
        print("\n❎ File was not saved.")

if __name__ == "__main__":
    # A bit more robust way to handle LMS column names from config
    # These will be used if they are defined in config.py, otherwise the defaults at the top are used.
    # This assumes LMS_CITY_COL and LMS_CLUSTER_COL are defined in config.py.
    # To avoid NameError if they are not in config, we can use getattr with default.
    # However, the try-except for config import should handle if they are missing.
    # For the socioeconomic feature creation:
    LMS_CITY_COL_CLEANED = "lms_city_col_cleaned_temp" # Temporary unique name for merge key
    main()