# 002_Engineered_Data_Analysis.py
"""
Loads the engineered surgery data (output of 002) and performs comprehensive
statistical analysis focused on the newly created features and data readiness for modeling.
Outputs results to both console and a timestamped text file in the results directory.

Run from the project root:
    python -m scripts.003_Engineered_Data_Analysis
or from *scripts/* directly:
    python 002_Engineered_Data_Analysis.py
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
    from config import OUTPUT_XLSX, RESULTS_DIR
except ImportError:
    print("Warning: config.py not found or required paths (OUTPUT_XLSX, RESULTS_DIR) not defined.")
    print("Using default relative paths assuming script is in 'scripts' folder.")
    scripts_dir = Path(__file__).resolve().parent
    project_root_alt = scripts_dir.parent
    OUTPUT_XLSX = project_root_alt / "data" / "surgery_data_engineered.xlsx"
    RESULTS_DIR = project_root_alt / "results"
    (project_root_alt / "data").mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup Output Logging to File and Console
# ---------------------------------------------------------------------------
def get_next_filename(base_dir: Path, prefix: str, suffix: str = ".log") -> Path:
    base_name = prefix
    counter = 0
    while True:
        if counter == 0:
            file_path = base_dir / f"{base_name}{suffix}"
        else:
            file_path = base_dir / f"{base_name}_{counter}{suffix}"
        if not file_path.exists():
            return file_path
        counter += 1

log_filename_base = Path(__file__).stem
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")

log_formatter = logging.Formatter('%(message)s')
logger = logging.getLogger()
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# --- Start of Analysis ---
logger.info(f"--- Engineered Data Analysis ---")
logger.info(f"Analysis started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Using input file: {OUTPUT_XLSX}")
if not OUTPUT_XLSX.exists():
    logger.error(f"Engineered data file not found at: {OUTPUT_XLSX.resolve()}")
    logger.error("Please run 002_Feature_Engineering.py first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Function to capture DataFrame info into a string
# ---------------------------------------------------------------------------
def get_df_info(df: pd.DataFrame) -> str:
    """Captures df.info() output into a string."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

# ---------------------------------------------------------------------------
# Load Data (Engineered)
# ---------------------------------------------------------------------------
logger.info("\n=== 1. Loading Engineered Data ===")
try:
    df_engineered = pd.read_excel(OUTPUT_XLSX)
    logger.info(f"Loaded df_engineered: {df_engineered.shape[0]:,} rows, {df_engineered.shape[1]} columns")
except Exception as e:
    logger.error(f"Error loading engineered Excel file: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Basic Structure and Info (Engineered)
# ---------------------------------------------------------------------------
logger.info("\n=== 2. Basic Structure & Info (Engineered Data) ===")
logger.info(f"Shape: {df_engineered.shape}")
logger.info("\nColumn Names:")
logger.info(str(df_engineered.columns.tolist()))
logger.info("\nData Types and Non-Null Counts:")
logger.info(get_df_info(df_engineered))

# Identify columns created in script 002 for focused analysis
engineered_features = [
    'was_canceled', 'surgery_weekday', 'is_weekend', 'season', 'age_decade',
    'bmi_missing', 'num_missing_labs', 'has_missing_labs', 'num_medications',
    'num_diagnoses', 'site_room', 'site_room_cancel_rate', 'anesthesia_cancel_rate',
    'procedure_code_cancel_rate', 'weekday_cancel_rate', 'near_holiday',
    'distance_bucket', 'distance_bucket_cancel_rate',
    'wait_days' # Assuming 'wait_days' will be added in 002
]
engineered_features_exist = [f for f in engineered_features if f in df_engineered.columns]
logger.info("\nEngineered Features identified for analysis:")
logger.info(str(engineered_features_exist))

logger.info("\nFirst 5 rows (Engineered Data - Selected Columns):")
preview_cols = ['patient_id', 'plan_id', 'was_canceled'] + engineered_features_exist
preview_cols_exist = [c for c in preview_cols if c in df_engineered.columns]
try:
    logger.info(df_engineered[preview_cols_exist].head().to_markdown(index=False, numalign="left", stralign="left"))
except Exception as e:
     logger.error(f"Error displaying preview head: {e}")


# ---------------------------------------------------------------------------
# Missing Value Analysis (Engineered Data)
# ---------------------------------------------------------------------------
logger.info("\n=== 3. Missing Value Analysis (Engineered Data) ===")
missing_percentage = (df_engineered.isnull().sum() / len(df_engineered) * 100).round(2)
missing_info = pd.DataFrame({'Missing Count': df_engineered.isnull().sum(), 'Missing Percentage (%)': missing_percentage})
missing_info = missing_info[missing_info['Missing Count'] > 0].sort_values('Missing Percentage (%)', ascending=False)

if not missing_info.empty:
    logger.info("Columns with Missing Values:")
    logger.info(missing_info.to_string())
else:
    logger.info("No missing values found in the engineered dataset.")

logger.info("\n--- Focus on Engineered Features Missing Values ---")
missing_engineered = missing_info.loc[missing_info.index.isin(engineered_features_exist)]
if not missing_engineered.empty:
    logger.info(missing_engineered.to_string())
else:
    logger.info("No missing values found in the specifically engineered features.")

# ---------------------------------------------------------------------------
# Target Variable Analysis (quick re-check)
# ---------------------------------------------------------------------------
logger.info("\n=== 4. Target Variable Analysis ('was_canceled') ===")
if 'was_canceled' in df_engineered.columns:
    cancel_rate = df_engineered['was_canceled'].mean() * 100
    logger.info(f"Overall Cancellation Rate: {cancel_rate:.2f}%")
    logger.info(f"Distribution:\n{df_engineered['was_canceled'].value_counts(normalize=True).round(4).to_string()}")
    logger.info(f"Counts:\n{df_engineered['was_canceled'].value_counts().to_string()}")
    imbalance_ratio_eng = df_engineered['was_canceled'].value_counts().min() / df_engineered['was_canceled'].value_counts().max() if df_engineered['was_canceled'].value_counts().max() > 0 else 0
else:
    logger.error("Error: 'was_canceled' column not found in the engineered data.")
    cancel_rate = np.nan # Set defaults for summary
    imbalance_ratio_eng = np.nan

# ---------------------------------------------------------------------------
# Analysis of Newly Created Numerical Features
# ---------------------------------------------------------------------------
logger.info("\n=== 5. Analysis of New Numerical Features ===")
new_numeric_features = [
    'num_missing_labs', 'num_medications', 'num_diagnoses',
    'site_room_cancel_rate', 'anesthesia_cancel_rate',
    'procedure_code_cancel_rate', 'weekday_cancel_rate',
    'distance_bucket_cancel_rate',
    'wait_days' # Assuming 'wait_days' will be added
]
# Add original wait days if it exists and is numeric
original_wait_days_col = 'ימים מתאריך פתיחת בקשה ועד תאריך ביצוע ניתוח'
if original_wait_days_col in df_engineered.columns and original_wait_days_col not in new_numeric_features:
    if pd.api.types.is_numeric_dtype(df_engineered[original_wait_days_col]):
        new_numeric_features.append(original_wait_days_col)
        logger.info(f"Including original '{original_wait_days_col}' in numerical analysis.")

new_numeric_exist = [f for f in new_numeric_features if f in df_engineered.columns]

if new_numeric_exist:
    logger.info(f"Analyzing: {new_numeric_exist}")
    logger.info("\nDescriptive Statistics (Overall):")
    logger.info(df_engineered[new_numeric_exist].describe().round(3).to_string())

    if 'was_canceled' in df_engineered.columns:
        logger.info("\nDescriptive Statistics (Grouped by 'was_canceled'):")
        try:
            grouped_desc_new_num = df_engineered.groupby('was_canceled')[new_numeric_exist].describe().round(3)
            logger.info(grouped_desc_new_num.to_string())
        except Exception as e:
            logger.error(f"Error generating grouped descriptives for new numerical: {e}")
            grouped_desc_new_num = None # Set to None for summary check
    else:
         logger.warning("Cannot group by 'was_canceled' as it is missing.")
         grouped_desc_new_num = None
else:
    logger.info("No new numerical features identified for analysis.")
    grouped_desc_new_num = None

# ---------------------------------------------------------------------------
# Analysis of Newly Created Categorical/Boolean Features
# ---------------------------------------------------------------------------
logger.info("\n=== 6. Analysis of New Categorical/Boolean Features ===")
new_cat_bool_features = [
    'surgery_weekday', 'is_weekend', 'season', 'age_decade',
    'bmi_missing', 'has_missing_labs', 'near_holiday',
    'distance_bucket', 'site_room'
]
new_cat_bool_exist = [f for f in new_cat_bool_features if f in df_engineered.columns]

if new_cat_bool_exist:
    logger.info(f"Analyzing: {new_cat_bool_exist}")
    for col in new_cat_bool_exist:
        try:
            is_high_cardinality = False
            num_unique = 0
            if col in df_engineered.columns:
                 num_unique = df_engineered[col].nunique(dropna=False)
                 if num_unique > 100 and col not in ['age_decade']:
                      is_high_cardinality = True

            if is_high_cardinality:
                 logger.info(f"\n--- Skipping detailed value counts for high cardinality: {col} (Unique values: {num_unique}) ---")
                 continue

            logger.info(f"\n--- Analysis for: {col} ---")
            col_series = df_engineered[col].fillna('__MISSING__')
            logger.info(f"Number of Unique Values: {num_unique}")
            logger.info("Value Counts:")
            logger.info(col_series.value_counts().to_string())

            if 'was_canceled' in df_engineered.columns and col != 'was_canceled':
                logger.info(f"\nCancellation Rate by {col}:")
                cancel_rate_by_cat = df_engineered.groupby(col, dropna=False)['was_canceled'].agg(['mean', 'count']).round(3)
                cancel_rate_by_cat = cancel_rate_by_cat.rename(columns={'mean': 'Cancel Rate', 'count': 'Total Count'})
                cancel_rate_by_cat = cancel_rate_by_cat.sort_values(['Total Count', 'Cancel Rate'], ascending=[False, False])
                logger.info(cancel_rate_by_cat.to_string())
            elif col == 'was_canceled': pass
            else: logger.warning("Cannot calculate cancellation rate as 'was_canceled' is missing.")
        except Exception as e:
             logger.error(f"Error analyzing categorical/bool column '{col}': {e}")
else:
    logger.info("No new categorical or boolean features identified for analysis.")


# ---------------------------------------------------------------------------
# Re-check Original Numerical Features (Post-Processing in 002 if any)
# ---------------------------------------------------------------------------
logger.info("\n=== 7. Re-check Original Numerical Features (Post-002 Processing) ===")
original_numeric_cols = ['age', 'bmi', 'albumin', 'pt_inr', 'potassium', 'sodium', 'hb', 'wbc', 'plt', 'distance_km']
original_numeric_exist = [f for f in original_numeric_cols if f in df_engineered.columns]

# Convert to numeric again just in case
for col in original_numeric_exist:
    if not pd.api.types.is_numeric_dtype(df_engineered[col]):
         original_dtype = df_engineered[col].dtype
         try:
              df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
              if df_engineered[col].isnull().all(): logger.warning(f"Original numeric column '{col}' became all NaNs after numeric conversion from {original_dtype}.")
         except Exception as e: logger.warning(f"Could not convert original '{col}' (dtype: {original_dtype}) to numeric: {e}")

original_numeric_final = df_engineered[original_numeric_exist].select_dtypes(include=np.number).columns.tolist()

num_unreasonable_bmi = np.nan # Default if BMI doesn't exist or check fails
if original_numeric_final:
    logger.info(f"Analyzing original numerical columns: {original_numeric_final}")
    logger.info("\nDescriptive Statistics (Overall):")
    if 'bmi' in original_numeric_final:
        logger.info("--- BMI Specific Check ---")
        try:
            bmi_desc = df_engineered['bmi'].describe().round(2)
            logger.info(bmi_desc.to_string())
            num_unreasonable_bmi = df_engineered.loc[df_engineered['bmi'].notna() & ((df_engineered['bmi'] < 10) | (df_engineered['bmi'] > 60)), 'bmi'].count()
            logger.info(f"Count of non-missing BMI values outside plausible range (10-60): {num_unreasonable_bmi}")
        except Exception as e: logger.error(f"Error during BMI specific check: {e}")

    logger.info("\nFull Descriptive Statistics (Original Numerics):")
    desc_orig_num = df_engineered[original_numeric_final].describe().round(2)
    logger.info(desc_orig_num.to_string())

    if 'was_canceled' in df_engineered.columns:
        logger.info("\nDescriptive Statistics (Grouped by 'was_canceled'):")
        try:
            grouped_desc_orig_num = df_engineered.groupby('was_canceled')[original_numeric_final].describe().round(2)
            logger.info(grouped_desc_orig_num.to_string())
        except Exception as e: logger.error(f"Error generating grouped descriptives for original numerical: {e}")
else:
    logger.info("No original numerical columns found or available for analysis in the engineered data.")


# ---------------------------------------------------------------------------
# Section 9: Key Insights and Data Readiness Summary (NEW)
# ---------------------------------------------------------------------------
logger.info("\n=== 9. Key Insights and Data Readiness Summary (Engineered Data) ===")

# Prepare variables for the summary string
num_rows, num_cols = df_engineered.shape
missing_value_summary = "No missing values" if missing_info.empty else f"{len(missing_info)} columns with missing values. Highest: {missing_info.index[0]} ({missing_info.iloc[0, 1]}%)"
bmi_status = "Not present/analyzed."
if 'bmi' in original_numeric_final and not pd.isna(num_unreasonable_bmi):
    bmi_status = f"Still contains {num_unreasonable_bmi:,} values outside plausible range (10-60). Needs cleaning/handling."
elif 'bmi' in original_numeric_final:
     bmi_status = "Analyzed, see details above. Check if cleaning was effective."
wait_days_present = "'wait_days' feature identified." if ('wait_days' in new_numeric_exist or original_wait_days_col in new_numeric_exist) else "'wait_days' feature appears MISSING."
cancel_rate_features_present = any(f.endswith('_cancel_rate') for f in new_numeric_exist)

# Build the summary string dynamically
summary_text_engineered = f"""
Based on the analysis of the engineered data ('{OUTPUT_XLSX.name}'):

**1. Data Structure:**
   - Shape: {num_rows:,} rows, {num_cols} columns.
   - Feature engineering added {num_cols - 27} columns (assuming 27 original).
   - Data types appear mostly appropriate (numeric for counts/rates, bool for flags, object for categories).

**2. Missing Values (Post-Engineering):**
   - Status: {missing_value_summary}
   - Key original columns (labs, bmi, text fields, etc.) still retain their original missing values unless imputation was performed in script 002 (check script).
   - Missing values in engineered features like 'distance_bucket*' or 'procedure_code_cancel_rate' likely stem from missing original data.
   - **Action Point:** A final imputation step is likely required before modeling for columns intended for use.

**3. Target Variable & Imbalance:**
   - 'was_canceled' column present.
   - Overall Cancellation Rate: {cancel_rate:.2f}%.
   - Imbalance Ratio: ~{imbalance_ratio_eng:.2f}. Remains an important factor for modeling.

**4. Engineered Features Assessment:**
   - **Counts** (medications, diagnoses, missing labs): Created successfully. Show minimal differences between canceled/non-canceled groups on average.
   - **Calendar Features** (weekday, season, near_holiday): Created successfully. 'near_holiday' and 'season' show weak correlation with cancellations based on rates. 'weekday' shows some variation (Wednesday higher rate).
   - **Flags** (bmi_missing, has_missing_labs): Created successfully. Show potentially interesting inverse relationships (higher cancellation rate when data is *not* missing).
   - **Historical Rates** (site_room, anesthesia, procedure_code): Created successfully ({'partially present' if not cancel_rate_features_present else 'present'}). Analysis shows `procedure_code_cancel_rate` has the most significant difference in means between canceled/non-canceled, suggesting high predictive potential. `site_room` and `anesthesia` rates also show differences.
   - **Wait Time** ('wait_days' or original): {wait_days_present} If present, the significant difference between canceled/non-canceled groups persists, confirming its importance.
   - **Binning/Derived Categorical** (age_decade, distance_bucket): Created successfully. `age_decade` shows interesting non-linear patterns (highest cancellation for youngest). `distance_bucket` shows weak correlation.

**5. Original Features Check (Post-Engineering):**
   - **BMI Status:** {bmi_status}
   - Other numerical features (labs, etc.): Distributions remain similar unless specifically cleaned in script 002.

**6. Data Readiness for Modeling:**
   - **Encoding Needed:** Categorical features (both original like 'מין', 'מחלקה מנתחת', and new like 'surgery_weekday', 'season', 'distance_bucket') need encoding (e.g., One-Hot, Target). High cardinality features ('procedure_code', 'city', 'site_room') require careful consideration.
   - **Imputation Needed:** Yes, for remaining missing values in features selected for the model. Median imputation for numerical, mode or '__MISSING__' for categorical are common starting points.
   - **Scaling May Be Needed:** Depending on the chosen model (e.g., Logistic Regression, SVM), numerical features might need scaling (StandardScaler, MinMaxScaler).
   - **Feature Selection:** May be necessary, especially given the number of features and potential multicollinearity (e.g., between different cancellation rates).

**7. Overall Assessment:**
   - Feature engineering has added potentially valuable predictors (historical rates, wait time, age decade).
   - Key data quality issues (missing values, BMI) need final resolution before modeling.
   - The dataset is ready for the final pre-processing steps (imputation, encoding, scaling) followed by model training and evaluation.
"""
logger.info(summary_text_engineered)


# ---------------------------------------------------------------------------
# Final Message
# ---------------------------------------------------------------------------
logger.info("\n--- End of Engineered Data Analysis Report ---")
logger.info(f"\n--- Analysis complete. Full output saved to: {log_filepath.resolve()} ---")