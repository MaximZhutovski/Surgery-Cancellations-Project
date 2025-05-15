# 004_Data_Final_Preprocessing.py
"""
Loads the engineered surgery data (from 002_v3), performs final preprocessing steps
(cleaning, type conversion, feature selection), saves an intermediate Excel of X,
then performs imputation, encoding, scaling, splits into train/test sets,
and saves the processed data for model training.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------------------------------------------------------
# Setup: Add project root to PYTHONPATH and import config
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    # Corrected import: Use ENGINEERED_DATA_XLSX as input
    from config import (ENGINEERED_DATA_XLSX, RESULTS_DIR, DATA_DIR, PLOT_DIR, MODEL_DIR,
                        X_TRAIN_PROCESSED_PATH, Y_TRAIN_PATH,
                        X_TEST_PROCESSED_PATH, Y_TEST_PATH,
                        PREPROCESSOR_PIPELINE_PATH, PROCESSED_FEATURE_NAMES_PATH,
                        PREPROCESSED_X_INTERMEDIATE_XLSX) # Path for intermediate Excel
    print("Successfully imported paths from config.py for 004")
except ImportError as e:
    print(f"CRITICAL (004): Error importing from config.py: {e}")
    # Fallback paths
    scripts_dir = Path(__file__).resolve().parent
    project_root_alt = scripts_dir.parent
    ENGINEERED_DATA_XLSX = project_root_alt / "data" / "surgery_data_engineered_v3.xlsx" # Fallback to correct new name
    RESULTS_DIR = project_root_alt / "results"; DATA_DIR = project_root_alt / "data"
    PLOT_DIR = project_root_alt / "plots"; MODEL_DIR = DATA_DIR / "models"
    print(f"Warning (004): Using fallback paths. ENGINEERED_DATA_XLSX set to: {ENGINEERED_DATA_XLSX}")
    (project_root_alt / "data").mkdir(parents=True, exist_ok=True); RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True) # Ensure these exist too
    X_TRAIN_PROCESSED_PATH = DATA_DIR / "004_X_train_processed.joblib" # Match config naming
    Y_TRAIN_PATH = DATA_DIR / "004_y_train.joblib"
    X_TEST_PROCESSED_PATH = DATA_DIR / "004_X_test_processed.joblib"
    Y_TEST_PATH = DATA_DIR / "004_y_test.joblib"
    PREPROCESSOR_PIPELINE_PATH = DATA_DIR / "004_preprocessor_pipeline.joblib"
    PROCESSED_FEATURE_NAMES_PATH = DATA_DIR / "004_processed_feature_names.joblib"
    PREPROCESSED_X_INTERMEDIATE_XLSX = DATA_DIR / "004_X_pre_pipeline.xlsx"
except Exception as e:
    print(f"CRITICAL (004): An unexpected error occurred during config import or path init: {e}")
    sys.exit(1)

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup Output Logging
# ---------------------------------------------------------------------------
def get_next_filename(base_dir: Path, prefix: str, suffix: str = ".log") -> Path:
    base_name = prefix; counter = 0
    while True:
        file_path = base_dir / (f"{base_name}{suffix}" if counter == 0 else f"{base_name}_{counter}{suffix}")
        if not file_path.exists(): return file_path
        counter += 1
log_filename_base = Path(__file__).stem # Should be 004_Data_Final_Preprocessing
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(); logger.handlers.clear(); logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setFormatter(log_formatter); logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); logger.addHandler(console_handler)

# --- Start of Script ---
logger.info(f"--- Data Final Preprocessing (Script {log_filename_base} - Post 002_v3) ---")
logger.info(f"Using input file: {ENGINEERED_DATA_XLSX}") # Using the correct variable from config

if not ENGINEERED_DATA_XLSX.exists():
    logger.error(f"Engineered data file not found at: {ENGINEERED_DATA_XLSX.resolve()}")
    logger.error("Please ensure 002_Feature_Engineering.py (v3) ran successfully and config.py is correct.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------
logger.info("\n=== 1. Loading Engineered Data (from 002_v3) ===")
try:
    try:
        df = pd.read_excel(ENGINEERED_DATA_XLSX, sheet_name="features_v3") # Ensure sheet name matches 002 output
    except ValueError:
        logger.warning("Sheet 'features_v3' not found, trying to load the first sheet.")
        df = pd.read_excel(ENGINEERED_DATA_XLSX, sheet_name=0)
    logger.info(f"Loaded engineered data: {df.shape[0]:,} rows, {df.shape[1]} columns")
except Exception as e:
    logger.error(f"Error loading engineered Excel file: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Feature Selection and Target Definition
# ---------------------------------------------------------------------------
logger.info("\n=== 2. Feature Selection and Target Definition ===")
# (הגדרת TARGET_COL, potential_numeric_features, potential_categorical_features, cols_to_drop_explicit
#  צריכה להיות כאן, כפי שהייתה בגרסה הקודמת והטובה של 004, בהתאם להמלצות מ-003)

TARGET_COL = 'was_canceled'
if TARGET_COL not in df.columns:
    logger.error(f"Target column '{TARGET_COL}' not found in the DataFrame!")
    sys.exit(1)
df[TARGET_COL] = df[TARGET_COL].astype(int)

potential_numeric_features = [
    'age', 'bmi', 'albumin', 'pt_inr', 'potassium', 'sodium', 'hb', 'wbc', 'plt', 'distance_km',
    'wait_days', 'days_to_next_holiday', 'days_from_prev_holiday',
    'num_existing_labs', 
    'num_medications', 'num_diagnoses',
    'site_room_cancel_rate_smoothed', 'anesthesia_cancel_rate_smoothed',
    'procedure_code_cancel_rate_smoothed', 'weekday_cancel_rate_smoothed',
    'distance_bucket_cancel_rate_smoothed',
    'is_weekend', 'bmi_missing', 'has_missing_labs', 'near_holiday', 'is_surgery_after_holiday_weekend'
]
potential_categorical_features = [
    'department', 'surgery_site', 'anesthesia', 'gender', 'payer', 'marital_status',
    'surgery_weekday', 'season', 'age_decade', 
    'wait_days_category', 'bmi_category', 'distance_bucket',
    'site_room'
]
cols_to_drop_explicit = [
    'patient_id', 'plan_id', 'request_date', 'surgery_date',
    'מספר ניתוח ביצוע', 'ימים מתאריך פתיחת בקשה ועד תאריך ביצוע ניתוח',
    'medications', 'diagnoses',
    'procedure_code', 'city', 'room', 'socioeconomic_cluster' 
]

# --- Data Type Conversions and Pre-split Cleaning (on df before X,y split) ---
logger.info("\n--- Pre-split Data Type Conversions & Cleaning ---")
# (כל קוד הניקוי וההמרות כאן נשאר זהה לגרסה הקודמת של 004)
bool_to_int_cols = ['is_weekend', 'bmi_missing', 'has_missing_labs', 'near_holiday', 'is_surgery_after_holiday_weekend']
for col in bool_to_int_cols:
    if col in df.columns:
        if df[col].dtype == 'bool': df[col] = df[col].astype(int)
        elif pd.api.types.is_numeric_dtype(df[col]): df[col] = df[col].astype(int)
        logger.info(f"Ensured boolean/flag feature '{col}' is int (0/1).")

if 'sodium' in df.columns and df['sodium'].dtype == 'object':
    df['sodium'] = pd.to_numeric(df['sodium'], errors='coerce')
    logger.info("Coerced 'sodium' to numeric.")
if 'bmi' in df.columns:
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce').clip(lower=10, upper=70)
    logger.info("Clipped 'bmi' to range [10, 70].")
if 'age_decade' in df.columns:
    df['age_decade'] = df['age_decade'].astype(str).replace({'<NA>': '__MISSING__', 'nan': '__MISSING__'})
    logger.info("Converted 'age_decade' to string, replaced <NA>/'nan' with '__MISSING__'.")
for col in potential_categorical_features:
    if col in df.columns and col != 'age_decade':
        if pd.api.types.is_categorical_dtype(df[col]):
            if '__MISSING__' not in df[col].cat.categories:
                try: df[col] = df[col].cat.add_categories('__MISSING__')
                except: df[col] = df[col].astype(object)
            df[col] = df[col].fillna('__MISSING__')
        else:
            df[col] = df[col].astype(str).fillna('__MISSING__')
        df.loc[df[col] == 'nan', col] = '__MISSING__'
        logger.info(f"Ensured '{col}' is string and NaNs (or 'nan' strings) are '__MISSING__'.")

y = df[TARGET_COL]
existing_cols_to_drop_final = [col for col in cols_to_drop_explicit if col in df.columns]
X = df.drop(columns=[TARGET_COL] + existing_cols_to_drop_final, errors='ignore')
numeric_features = [f for f in potential_numeric_features if f in X.columns and pd.api.types.is_numeric_dtype(X[f])]
categorical_features = [f for f in potential_categorical_features if f in X.columns]
all_selected_features = list(set(numeric_features + categorical_features))
X = X[all_selected_features].copy()

logger.info(f"Shape of X after selection & pre-split cleaning: {X.shape}, Shape of y: {y.shape}")
logger.info(f"Final Numeric Features for pipeline ({len(numeric_features)}): {numeric_features}")
logger.info(f"Final Categorical Features for pipeline ({len(categorical_features)}): {categorical_features}")

logger.info(f"\n=== Saving Intermediate X (before split and pipeline) to Excel: {PREPROCESSED_X_INTERMEDIATE_XLSX} ===")
try:
    X.to_excel(PREPROCESSED_X_INTERMEDIATE_XLSX, index=False, engine='openpyxl')
    logger.info(f"Successfully saved X to {PREPROCESSED_X_INTERMEDIATE_XLSX}")
except Exception as e:
    logger.error(f"Error saving intermediate X to Excel: {e}")

# ---------------------------------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------------------------------
logger.info("\n=== 3. Train-Test Split ===")
# (קוד החלוקה נשאר זהה)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ---------------------------------------------------------------------------
# 4. Preprocessing Pipelines Definition
# ---------------------------------------------------------------------------
logger.info("\n=== 4. Preprocessing Pipelines Definition ===")
# (קוד הגדרת ה-pipelines וה-ColumnTransformer נשאר זהה)
numeric_pipeline = Pipeline([('imputer_num', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
categorical_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, min_frequency=10))])
final_numeric_for_pipeline = [f for f in numeric_features if f in X_train.columns]
final_categorical_for_pipeline = [f for f in categorical_features if f in X_train.columns]
logger.info(f"Numeric features for ColumnTransformer ({len(final_numeric_for_pipeline)}): {final_numeric_for_pipeline}")
logger.info(f"Categorical features for ColumnTransformer ({len(final_categorical_for_pipeline)}): {final_categorical_for_pipeline}")
transformers_list = []
if final_numeric_for_pipeline: transformers_list.append(('num', numeric_pipeline, final_numeric_for_pipeline))
if final_categorical_for_pipeline: transformers_list.append(('cat', categorical_pipeline, final_categorical_for_pipeline))
preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop')

# ---------------------------------------------------------------------------
# 5. Applying Preprocessor and Saving Processed Data
# ---------------------------------------------------------------------------
# (קוד החלת ה-preprocessor ושמירת הנתונים נשאר זהה)
logger.info("\n=== 5. Applying Preprocessor and Saving Data ===")
X_train_processed = preprocessor.fit_transform(X_train); logger.info(f"X_train_processed shape: {X_train_processed.shape}")
X_test_processed = preprocessor.transform(X_test); logger.info(f"X_test_processed shape: {X_test_processed.shape}")
try:
    ohe_feature_names = []
    ct_transformers = preprocessor.named_transformers_
    if 'cat' in ct_transformers and hasattr(ct_transformers['cat'], 'get_feature_names_out'):
         ohe_feature_names = ct_transformers['cat'].get_feature_names_out(final_categorical_for_pipeline)
    processed_feature_names_list = list(final_numeric_for_pipeline) + list(ohe_feature_names)
    if len(processed_feature_names_list) != X_train_processed.shape[1]:
        logger.warning(f"Name-column mismatch. Using generic names."); processed_feature_names_list = [f"feat_{i}" for i in range(X_train_processed.shape[1])]
    logger.info(f"Number of features after OHE & processing: {len(processed_feature_names_list)}")
except Exception as e:
    logger.warning(f"Could not get OHE names: {e}. Using generic."); processed_feature_names_list = [f"feat_{i}" for i in range(X_train_processed.shape[1])]
try:
    joblib.dump(X_train_processed, X_TRAIN_PROCESSED_PATH); logger.info(f"Saved X_train_processed to: {X_TRAIN_PROCESSED_PATH}")
    joblib.dump(y_train, Y_TRAIN_PATH); logger.info(f"Saved y_train to: {Y_TRAIN_PATH}")
    joblib.dump(X_test_processed, X_TEST_PROCESSED_PATH); logger.info(f"Saved X_test_processed to: {X_TEST_PROCESSED_PATH}")
    joblib.dump(y_test, Y_TEST_PATH); logger.info(f"Saved y_test to: {Y_TEST_PATH}")
    joblib.dump(preprocessor, PREPROCESSOR_PIPELINE_PATH); logger.info(f"Saved preprocessor to: {PREPROCESSOR_PIPELINE_PATH}")
    if processed_feature_names_list:
        joblib.dump(processed_feature_names_list, PROCESSED_FEATURE_NAMES_PATH); logger.info(f"Saved feature names to: {PROCESSED_FEATURE_NAMES_PATH}")
except Exception as e: logger.error(f"Error saving data: {e}"); sys.exit(1)

logger.info(f"\n--- Data Final Preprocessing (Script {log_filename_base}) complete ---")
logger.info(f"Processed files saved in: {DATA_DIR.resolve()}")
logger.info(f"\n--- Full output saved to: {log_filepath.resolve()} ---")

if __name__ == "__main__":
    pass