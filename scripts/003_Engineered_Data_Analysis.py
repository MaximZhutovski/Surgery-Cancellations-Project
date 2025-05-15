# 003_Engineered_Data_Analysis.py
"""
Loads the engineered surgery data (output of 002_Feature_Engineering.py v3)
and performs comprehensive statistical analysis focused on all features
and data readiness for modeling. Outputs results to both console and a
timestamped text file in the results directory.
"""
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import logging

# ---------------------------------------------------------------------------
# Setup: Add project root to PYTHONPATH and import config
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import ENGINEERED_DATA_XLSX, RESULTS_DIR, COLS_MAP
    print("Successfully imported paths from config.py for 003")
except ImportError:
    print("CRITICAL (003): config.py not found or essential paths (ENGINEERED_DATA_XLSX, RESULTS_DIR, COLS_MAP) missing.")
    scripts_dir = Path(__file__).resolve().parent
    project_root_alt = scripts_dir.parent
    ENGINEERED_DATA_XLSX = project_root_alt / "data" / "surgery_data_engineered_v3.xlsx"
    RESULTS_DIR = project_root_alt / "results"
    COLS_MAP = {}
    print(f"Warning (003): Using fallback paths. ENGINEERED_DATA_XLSX set to: {ENGINEERED_DATA_XLSX}")
    (project_root_alt / "data").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"CRITICAL (003): An unexpected error occurred during config import: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Setup Output Logging
# ---------------------------------------------------------------------------
def get_next_filename(base_dir: Path, prefix: str, suffix: str = ".log") -> Path:
    base_name = prefix; counter = 0
    while True:
        file_path = base_dir / (f"{base_name}{suffix}" if counter == 0 else f"{base_name}_{counter}{suffix}")
        if not file_path.exists(): return file_path
        counter += 1

log_filename_base = Path(__file__).stem
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(); logger.handlers.clear(); logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setFormatter(log_formatter); logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); logger.addHandler(console_handler)

# --- Start of Analysis ---
logger.info(f"--- Engineered Data Analysis (Script {log_filename_base} - Post 002_v3) ---")
logger.info(f"Analysis started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Using input file: {ENGINEERED_DATA_XLSX} (output of 002_Feature_Engineering.py v3)")
if not ENGINEERED_DATA_XLSX.exists():
    logger.error(f"Engineered data file not found at: {ENGINEERED_DATA_XLSX.resolve()}")
    logger.error("Please run the updated 002_Feature_Engineering.py (v3) first and ensure config.py points to its output.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Function to capture DataFrame info into a string
# ---------------------------------------------------------------------------
def get_df_info(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.info(buf=buffer, verbose=True, show_counts=True)
    return buffer.getvalue()

# ---------------------------------------------------------------------------
# Load Data (Engineered)
# ---------------------------------------------------------------------------
logger.info("\n=== 1. Loading Engineered Data ===")
try:
    try:
        df_engineered = pd.read_excel(ENGINEERED_DATA_XLSX, sheet_name="features_v3")
    except ValueError:
        logger.warning("Sheet 'features_v3' not found, trying to load the first sheet.")
        df_engineered = pd.read_excel(ENGINEERED_DATA_XLSX, sheet_name=0)
    logger.info(f"Loaded df_engineered: {df_engineered.shape[0]:,} rows, {df_engineered.shape[1]} columns")
except Exception as e:
    logger.error(f"Error loading engineered Excel file: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Basic Structure and Info
# ---------------------------------------------------------------------------
logger.info("\n=== 2. Basic Structure & Info (Engineered Data) ===")
logger.info(f"Shape: {df_engineered.shape}")
logger.info("\nColumn Names and Dtypes (from df.info()):")
logger.info(get_df_info(df_engineered))

# Identify original, newly created, and flag features for focused analysis
original_features_renamed = [ # From COLS_MAP and other known originals
    'patient_id', 'request_date', 'surgery_date', 'plan_id',
    'department', 'surgery_site', 'room', 'procedure_code', 'anesthesia',
    'age', 'gender', 'city', 'payer', 'marital_status', 'bmi',
    'medications', 'diagnoses',
    'sodium', 'potassium', 'albumin', 'hb', 'wbc', 'plt', 'pt_inr', 'distance_km'
]
engineered_numeric_features = [ # Numeric features created or mainly processed in 002
    'wait_days', 'days_to_next_holiday', 'days_from_prev_holiday',
    'num_existing_labs', 'num_medications', 'num_diagnoses',
    'site_room_cancel_rate_smoothed', 'anesthesia_cancel_rate_smoothed',
    'procedure_code_cancel_rate_smoothed', 'weekday_cancel_rate_smoothed',
    'distance_bucket_cancel_rate_smoothed',
    'socioeconomic_cluster' # (Will be all NaN if LMS file failed)
]
engineered_categorical_features = [ # Categorical features created or mainly processed in 002
    'surgery_weekday', 'season', 'age_decade', # age_decade often treated as categorical
    'wait_days_category', 'bmi_category', 'distance_bucket',
    'site_room'
]
engineered_boolean_flags = [ # Boolean flags created in 002
    'was_canceled', 'is_weekend', 'bmi_missing', 'has_missing_labs',
    'near_holiday', 'is_surgery_after_holiday_weekend'
]

logger.info("\nFirst 5 rows (Sample of Engineered Data - Selected Columns):")
preview_cols_sample = [
    'plan_id', 'was_canceled', 'wait_days', 'wait_days_category', 'department', 'age_decade', 'bmi_category',
    'socioeconomic_cluster', 'days_to_next_holiday', 'is_surgery_after_holiday_weekend',
    'procedure_code_cancel_rate_smoothed', 'distance_bucket'
]
preview_cols_exist = [c for c in preview_cols_sample if c in df_engineered.columns]
if preview_cols_exist:
    try:
        logger.info(df_engineered[preview_cols_exist].head().to_markdown(index=False, numalign="left", stralign="left"))
    except Exception as e:
        logger.error(f"Error displaying preview head (markdown): {e}. Showing basic head:")
        logger.info(df_engineered[preview_cols_exist].head())
else:
    logger.warning("No preview columns available. Showing all available from first 5 rows.")
    try:
        logger.info(df_engineered.head().to_markdown(index=False, numalign="left", stralign="left"))
    except Exception as e:
        logger.info(f"Error showing full head: {e}")
        logger.info(df_engineered.head())

# ---------------------------------------------------------------------------
# Missing Value Analysis (This is Section 3)
# ---------------------------------------------------------------------------
logger.info("\n=== 3. Missing Value Analysis (Engineered Data) ===")
missing_counts = df_engineered.isnull().sum()
missing_percentage = (missing_counts / len(df_engineered) * 100).round(2)
missing_info = pd.DataFrame({'Missing Count': missing_counts, 'Missing Percentage (%)': missing_percentage}) # Definition of missing_info
missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values('Missing Percentage (%)', ascending=False)

if not missing_info.empty:
    logger.info("Columns with Missing Values:")
    logger.info(missing_info.to_string())
else:
    logger.info("No missing values found in the engineered dataset (after 002 processing).")

# ---------------------------------------------------------------------------
# Target Variable Analysis
# ---------------------------------------------------------------------------
logger.info("\n=== 4. Target Variable Analysis ('was_canceled') ===")
if 'was_canceled' in df_engineered.columns:
    df_engineered['was_canceled'] = df_engineered['was_canceled'].astype(bool)
    cancel_rate = df_engineered['was_canceled'].mean() * 100
    logger.info(f"Overall Cancellation Rate: {cancel_rate:.2f}%")
    logger.info(f"Distribution:\n{df_engineered['was_canceled'].value_counts(normalize=True).round(4).to_string()}")
else:
    logger.error("Critical: 'was_canceled' column not found."); sys.exit(1)

# ---------------------------------------------------------------------------
# Numerical Feature Analysis (Original and New)
# ---------------------------------------------------------------------------
logger.info("\n=== 5. Numerical Feature Analysis ===")
# Combine all potential numeric features, including originals that should still be numeric
# and newly created numeric features.
potential_numeric_for_analysis_s5 = [
    'age', 'bmi', 'albumin', 'pt_inr', 'potassium', 'sodium', 'hb', 'wbc', 'plt', 'distance_km'
]
potential_numeric_for_analysis_s5.extend([col for col in engineered_numeric_features if col in df_engineered.columns])
# Boolean flags will be converted to 0/1 in 004 and treated as numeric there,
# but for this EDA, we'll analyze them separately in section 7.
potential_numeric_for_analysis_s5 = list(set(potential_numeric_for_analysis_s5) - set(engineered_boolean_flags))


numeric_cols_final_s5 = []
for col in potential_numeric_for_analysis_s5:
    if col in df_engineered.columns:
        if pd.api.types.is_numeric_dtype(df_engineered[col]):
            numeric_cols_final_s5.append(col)
        else: # Attempt conversion for columns that might be object but should be numeric
            try:
                df_engineered[f"{col}_numeric_temp"] = pd.to_numeric(df_engineered[col], errors='coerce')
                if df_engineered[f"{col}_numeric_temp"].isnull().all() and df_engineered[col].notnull().any():
                    logger.warning(f"Column '{col}' became all NaNs after coercive numeric conversion. Original dtype: {df_engineered[col].dtype}")
                elif pd.api.types.is_numeric_dtype(df_engineered[f"{col}_numeric_temp"]):
                    df_engineered[col] = df_engineered[f"{col}_numeric_temp"] # Replace original if conversion successful
                    numeric_cols_final_s5.append(col)
                    logger.info(f"Successfully converted column '{col}' to numeric for analysis.")
                else:
                    logger.warning(f"Could not convert '{col}' to numeric (current: {df_engineered[col].dtype}). Will be skipped in numeric analysis.")
                if f"{col}_numeric_temp" in df_engineered.columns:
                    df_engineered.drop(columns=[f"{col}_numeric_temp"], inplace=True) # Clean up temp column
            except Exception as e:
                logger.warning(f"Error converting column '{col}' to numeric: {e}. Will be skipped.")

logger.info(f"Identified numerical columns for analysis ({len(numeric_cols_final_s5)}): {numeric_cols_final_s5}")
if numeric_cols_final_s5:
    logger.info("\nDescriptive Statistics (Overall - Numerical Features):")
    logger.info(df_engineered[numeric_cols_final_s5].describe().round(3).to_string())
    logger.info("\nDescriptive Statistics (Grouped by 'was_canceled' - Numerical Features):")
    try:
        grouped_desc_num = df_engineered.groupby('was_canceled')[numeric_cols_final_s5].describe().round(3)
        logger.info(grouped_desc_num.to_string())
    except Exception as e:
        logger.error(f"Error generating grouped descriptives for numerical features: {e}")
else:
    logger.info("No numerical features available for detailed analysis after type checks.")

if 'bmi' in numeric_cols_final_s5: # Check if 'bmi' is in the list of numeric columns
    logger.info("\n--- BMI Specific Check (Engineered Data) ---")
    unreasonable_bmi_count = df_engineered.loc[df_engineered['bmi'].notna() & ((df_engineered['bmi'] < 10) | (df_engineered['bmi'] > 70)), 'bmi'].count()
    logger.info(f"Count of non-missing BMI values outside plausible range (10-70): {unreasonable_bmi_count}")
    if unreasonable_bmi_count > 0: logger.warning("BMI still contains values outside 10-70. Cleaning in 004 is essential.")

# ---------------------------------------------------------------------------
# Categorical Feature Analysis (Original and New)
# ---------------------------------------------------------------------------
logger.info("\n=== 6. Categorical Feature Analysis ===")
# Combine original and new categorical features
potential_categorical_for_analysis_s6 = [
    'department', 'surgery_site', 'room', 'procedure_code', 'anesthesia',
    'gender', 'city', 'payer', 'marital_status' # Originals
]
potential_categorical_for_analysis_s6.extend([col for col in engineered_categorical_features if col in df_engineered.columns])
if 'age_decade' in df_engineered.columns: # age_decade is numeric but often analyzed as categorical
    # Ensure it's treated as object/string for this part if it's numeric
    df_engineered['age_decade_cat_analysis'] = df_engineered['age_decade'].astype(str).replace({'<NA>': '__MISSING__', 'nan': '__MISSING__'})
    potential_categorical_for_analysis_s6.append('age_decade_cat_analysis')

potential_categorical_for_analysis_s6 = list(set(potential_categorical_for_analysis_s6))
categorical_cols_final_s6 = [col for col in potential_categorical_for_analysis_s6 if col in df_engineered.columns]

logger.info(f"Identified categorical columns for analysis ({len(categorical_cols_final_s6)}): {categorical_cols_final_s6}")
for col in categorical_cols_final_s6:
    if col not in df_engineered.columns: continue
    logger.info(f"\n--- Analysis for Categorical Feature: {col} ---")
    
    # Use a copy for manipulation for this specific column analysis
    current_col_series = df_engineered[col].copy()
    if pd.api.types.is_categorical_dtype(current_col_series):
        if '__MISSING__' not in current_col_series.cat.categories:
            try: current_col_series = current_col_series.cat.add_categories('__MISSING__')
            except: current_col_series = current_col_series.astype(object) # Fallback to object
        current_col_series.fillna('__MISSING__', inplace=True)
    else: # If not categorical, convert to string and fill
        current_col_series = current_col_series.astype(str).fillna('__MISSING__')
    
    # Ensure literal 'nan' strings are also treated as '__MISSING__'
    current_col_series.replace('nan', '__MISSING__', inplace=True)

    num_unique = current_col_series.nunique()
    logger.info(f"Number of Unique Values: {num_unique}")
    
    # Limit printing for very high cardinality features
    # Adjust threshold as needed, e.g. > 50 if many have 30-50 unique values.
    # Exclude 'procedure_code', 'city', 'site_room' from full value_counts if too long.
    print_limit = 15
    high_card_exceptions = ['procedure_code', 'city', 'site_room']
    if num_unique > 30 and col not in high_card_exceptions:
        logger.info(f"Top {print_limit} Most Frequent Values for '{col}':")
        logger.info(current_col_series.value_counts().head(print_limit).to_string())
    elif col in high_card_exceptions and num_unique > print_limit :
         logger.info(f"Top {print_limit} Most Frequent Values for '{col}' (High Cardinality):")
         logger.info(current_col_series.value_counts().head(print_limit).to_string())
    else:
        logger.info(f"Value Counts for '{col}':")
        logger.info(current_col_series.value_counts().to_string())

    logger.info(f"\nCancellation Rate by {col}:")
    try:
        # Group by the modified series that has NaNs handled as '__MISSING__'
        temp_df_for_groupby = pd.DataFrame({'category_col': current_col_series, 'was_canceled': df_engineered['was_canceled']})
        cancel_rate_by_cat = temp_df_for_groupby.groupby('category_col')['was_canceled'].agg(['mean', 'count']).round(3)
        cancel_rate_by_cat = cancel_rate_by_cat.rename(columns={'mean': 'Cancel Rate', 'count': 'Total Count'})
        cancel_rate_by_cat = cancel_rate_by_cat.sort_values(['Total Count', 'Cancel Rate'], ascending=[False, False])
        
        if num_unique > 30 and col not in high_card_exceptions:
             logger.info(cancel_rate_by_cat.head(print_limit).to_string())
             logger.info("... (showing top 15 by count for high cardinality features)")
        elif col in high_card_exceptions and num_unique > print_limit:
             logger.info(cancel_rate_by_cat.head(print_limit).to_string())
             logger.info("... (showing top 15 by count for high cardinality features)")
        else:
            logger.info(cancel_rate_by_cat.to_string())
    except Exception as e:
        logger.error(f"Error calculating cancellation rate for '{col}': {e}")

if 'age_decade_cat_analysis' in df_engineered.columns: # Clean up temp column
    df_engineered.drop(columns=['age_decade_cat_analysis'], inplace=True)

# ---------------------------------------------------------------------------
# Boolean Flag Analysis
# ---------------------------------------------------------------------------
logger.info("\n=== 7. Boolean Flag Analysis ===")
# Ensure 'was_canceled' itself is not in this list for analysis
boolean_flags_final_s7 = [col for col in engineered_boolean_flags if col in df_engineered.columns and col != 'was_canceled']
logger.info(f"Identified boolean flags for analysis ({len(boolean_flags_final_s7)}): {boolean_flags_final_s7}")

for flag in boolean_flags_final_s7:
    if flag not in df_engineered.columns: continue
    logger.info(f"\n--- Analysis for Boolean Flag: {flag} ---")
    # Make a copy for manipulation
    flag_series_for_analysis = df_engineered[flag].copy()
    if flag_series_for_analysis.isnull().any() or not pd.api.types.is_bool_dtype(flag_series_for_analysis):
        logger.warning(f"Flag '{flag}' contains NaNs or is not boolean (dtype: {flag_series_for_analysis.dtype}). Attempting conversion/fillna with False.")
        flag_series_for_analysis = pd.to_numeric(flag_series_for_analysis, errors='coerce').fillna(0).astype(bool) # Fill NaNs with False

    logger.info(f"Value Counts for '{flag}':")
    logger.info(flag_series_for_analysis.value_counts().to_string())
    logger.info(f"Cancellation Rate by {flag}:")
    try:
        temp_df_for_flag_groupby = pd.DataFrame({'flag_col': flag_series_for_analysis, 'was_canceled': df_engineered['was_canceled']})
        flag_cancel_rates = temp_df_for_flag_groupby.groupby('flag_col')['was_canceled'].agg(['mean', 'count']).round(3)
        flag_cancel_rates = flag_cancel_rates.rename(columns={'mean': 'Cancel Rate', 'count': 'Total Count'})
        logger.info(flag_cancel_rates.to_string())
    except Exception as e:
        logger.error(f"Error calculating cancellation rate for flag '{flag}': {e}")

# ---------------------------------------------------------------------------
# Section 8: Data Readiness Summary and Recommendations for 004
# ---------------------------------------------------------------------------
logger.info("\n=== 8. Data Readiness Summary and Recommendations for 004_Data_Final_Preprocessing ===")
# (This section uses missing_info from Section 3, and other variables defined globally in this script)
num_rows_s8, num_cols_s8 = df_engineered.shape # Use specific names for clarity
missing_value_summary_eng_s8 = "No missing values found that require special handling beyond imputation/flags."
if 'missing_info' in locals() and not missing_info.empty: # Check if missing_info was defined
    top_missing_col_s8 = missing_info.index[0]
    top_missing_pct_s8 = missing_info.iloc[0, 1]
    missing_value_summary_eng_s8 = f"{len(missing_info)} columns with missing values. Highest: {top_missing_col_s8} ({top_missing_pct_s8}%)"

socio_status_s8 = "'socioeconomic_cluster' was not successfully created or is all NaN."
if 'socioeconomic_cluster' in df_engineered.columns and df_engineered['socioeconomic_cluster'].notna().any():
    # Ensure it's numeric for min/max
    temp_socio = pd.to_numeric(df_engineered['socioeconomic_cluster'], errors='coerce')
    if temp_socio.notna().any() and pd.api.types.is_numeric_dtype(temp_socio):
        socio_status_s8 = f"Created and numeric. Min: {temp_socio.min():.2f}, Max: {temp_socio.max():.2f}."
    elif temp_socio.notna().any():
        socio_status_s8 = f"Created but not purely numeric (has non-coercible values). Dtype: {df_engineered['socioeconomic_cluster'].dtype}."
    # If after coerce it's all NaN but original had some values, it means they weren't numeric.

summary_text_recommendations = f"""
**1. Data Structure:**
   - Shape: {num_rows_s8:,} rows, {num_cols_s8} columns. Many new features from 002_v3.

**2. Missing Values (Post-002 v3):**
   - Status: {missing_value_summary_eng_s8} (Full list in Section 3 output).
   - 'socioeconomic_cluster' status: {socio_status_s8} (Based on 002, this is likely all NaN due to LMS file issue).
   - Other new categorical features like 'wait_days_category', 'bmi_category', 'distance_bucket' have '__MISSING__' for original NaNs.
   - **Action for 004:** Standard imputation for numeric features. Categorical features with '__MISSING__' will be a distinct category for OHE.

**3. Target Variable ('was_canceled'):**
   - Rate: {df_engineered['was_canceled'].mean()*100:.2f}% canceled. Imbalance handling is key.

**4. Feature Assessment & Recommendations for 004 (Selection & Encoding):**

   **A. Potential Numeric Features for Model (from {len(numeric_cols_final_s5) if 'numeric_cols_final_s5' in locals() else 'N/A'} identified in Sec 5):**
      - Core Numerics: `age`, `bmi` (needs clip/impute), Lab tests (impute), `distance_km` (impute), `wait_days`.
      - New Time-based: `days_to_next_holiday`, `days_from_prev_holiday`.
      - New Counts: `num_existing_labs`, `num_medications`, `num_diagnoses`.
      - Smoothed Rates: All `_cancel_rate_smoothed` features. These are likely strong.
      - `socioeconomic_cluster`: Drop if all NaN. If fixed and numeric, keep.
      - Boolean flags (to be 0/1 in 004): `is_weekend`, `bmi_missing`, `has_missing_labs`, `near_holiday`, `is_surgery_after_holiday_weekend`.

   **B. Potential Categorical Features for Model (from {len(categorical_cols_final_s6) if 'categorical_cols_final_s6' in locals() else 'N/A'} identified in Sec 6 - for OHE):**
      - Originals (ensure they are in `COLS_MAP` and thus have English names if used):
         - `department` (Unique: {df_engineered['department'].nunique() if 'department' in df_engineered else 'N/A'}) - High card. Consider Target Encoding or using rate.
         - `surgery_site` (Unique: {df_engineered['surgery_site'].nunique() if 'surgery_site' in df_engineered else 'N/A'}) - Moderate card. OHE fine.
         - `anesthesia` (Unique: {df_engineered['anesthesia'].nunique() if 'anesthesia' in df_engineered else 'N/A'}) - Manageable. OHE fine.
         - `gender` (Unique: {df_engineered['gender'].nunique() if 'gender' in df_engineered else 'N/A'}) - Low card. OHE fine.
         - `payer` (Unique: {df_engineered['payer'].nunique() if 'payer' in df_engineered else 'N/A'}) - Moderate card. OHE fine.
         - `marital_status` (Unique: {df_engineered['marital_status'].nunique() if 'marital_status' in df_engineered else 'N/A'}) - Low card. OHE fine.
      - New/Processed Categoricals:
         - `surgery_weekday` (Low card. OHE fine)
         - `season` (Low card. OHE fine)
         - `age_decade` (as string, Low-ish card. OHE fine)
         - `wait_days_category` (Low card. OHE fine)
         - `bmi_category` (Low card. OHE fine)
         - `distance_bucket` (Low card. OHE fine)
         - `site_room` (Unique: {df_engineered['site_room'].nunique() if 'site_room' in df_engineered else 'N/A'}) - Potentially high card. If `site_room_cancel_rate_smoothed` is used, `site_room` itself might be dropped or target encoded.

   **C. Features to Likely Drop/Handle Specially in 004:**
      - Identifiers: `patient_id`, `plan_id`.
      - Raw Dates: `request_date`, `surgery_date`.
      - Original Text: `medications`, `diagnoses`.
      - Very High Cardinality (if not Target Encoded):
         - `procedure_code`: Use `procedure_code_cancel_rate_smoothed` instead. Drop raw.
         - `city`: Drop raw. (Relay on distance features; socio-economic if it worked).
      - Redundant components: `room` (if `site_room` used). `surgery_site` could also be dropped if `site_room` and its rate are very effective.
      - All-NaN features: `socioeconomic_cluster` (currently).
      - Any remaining original Hebrew columns not mapped via `COLS_MAP` (e.g., "ימים מתאריך פתיחת בקשה ועד תאריך ביצוע ניתוח").

**5. Preprocessing in 004:**
   - **Numeric:** Imputation (median), Scaling (StandardScaler).
   - **Categorical:** Ensure `__MISSING__` is a category, then OneHotEncoder (with `handle_unknown='ignore'`, possibly `min_frequency`).
      For very high cardinality features not dropped, **Target Encoding with cross-validation** is a strong alternative to OHE.
   - **Boolean flags:** Convert to 0/1 and include in numeric pipeline.

**6. Overall Readiness:**
   - Dataset is significantly enriched. Focus in 004 should be on selecting the most promising subset of these features, managing dimensionality from OHE, and robustly handling any remaining data quality issues before modeling.
"""
logger.info(summary_text_recommendations)

logger.info(f"\n--- End of Engineered Data Analysis Report (Script {log_filename_base}) ---")
logger.info(f"\n--- Full output saved to: {log_filepath.resolve()} ---")

# Define a main function to encapsulate the script logic for cleaner execution
def main_eda():
    # All the logging and df_engineered manipulation happens here.
    # This function is implicitly called if the script is run directly.
    # No specific action needed here as the script runs sequentially.
    pass

if __name__ == "__main__":
    # No explicit main_eda() call needed as script runs top-to-bottom.
    # If we had structured the above into functions, we would call the main orchestrating function here.
    pass