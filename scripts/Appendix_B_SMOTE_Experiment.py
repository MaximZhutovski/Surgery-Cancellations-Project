# Appendix_B_SMOTE_Experiment.py
"""
Trains an XGBoost model using SMOTE for handling class imbalance.
Compares results with the previously tuned model.

Run from the project root:
    python -m scripts.007_Model_XGB_With_SMOTE
or from *scripts/* directly:
    python 007_Model_XGB_With_SMOTE.py
"""
import sys
import datetime
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline # Renamed to avoid conflict
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Setup: Add project root to PYTHONPATH and import config
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import (RESULTS_DIR, DATA_DIR, PLOT_DIR, MODEL_DIR,
                        X_TRAIN_PROCESSED_PATH, Y_TRAIN_PATH,
                        X_TEST_PROCESSED_PATH, Y_TEST_PATH,
                        PROCESSED_FEATURE_NAMES_PATH,
                        BEST_XGB_PARAMS_PATH) # Ensure this is in config

except ImportError as e:
    print(f"Error importing from config: {e}")
    print("Using default relative paths assuming script is in 'scripts' folder.")
    scripts_dir = Path(__file__).resolve().parent
    project_root_alt = scripts_dir.parent
    RESULTS_DIR = project_root_alt / "results"
    DATA_DIR = project_root_alt / "data"
    PLOT_DIR = project_root_alt / "plots"
    MODEL_DIR = DATA_DIR / "models"

    X_TRAIN_PROCESSED_PATH = DATA_DIR / "X_train_processed.joblib"
    Y_TRAIN_PATH = DATA_DIR / "y_train.joblib"
    X_TEST_PROCESSED_PATH = DATA_DIR / "X_test_processed.joblib"
    Y_TEST_PATH = DATA_DIR / "y_test.joblib"
    PROCESSED_FEATURE_NAMES_PATH = DATA_DIR / "processed_feature_names.joblib"
    BEST_XGB_PARAMS_PATH = RESULTS_DIR / "006_best_xgb_params_tuned.joblib"

    for d_path in (DATA_DIR, PLOT_DIR, RESULTS_DIR, MODEL_DIR):
        d_path.mkdir(parents=True, exist_ok=True)

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

# --- Start of Script ---
logger.info(f"--- XGBoost with SMOTE (Script 007) ---")

# ---------------------------------------------------------------------------
# 1. Load Processed Data and Best Hyperparameters
# ---------------------------------------------------------------------------
logger.info("=== 1. Loading Processed Data and Best Hyperparameters ===")
try:
    X_train_processed = joblib.load(X_TRAIN_PROCESSED_PATH)
    y_train = joblib.load(Y_TRAIN_PATH)
    X_test_processed = joblib.load(X_TEST_PROCESSED_PATH)
    y_test = joblib.load(Y_TEST_PATH)
    processed_feature_names = joblib.load(PROCESSED_FEATURE_NAMES_PATH)
    
    if BEST_XGB_PARAMS_PATH.exists():
        best_xgb_params = joblib.load(BEST_XGB_PARAMS_PATH)
        logger.info("Loaded best XGBoost hyperparameters from script 006.")
        logger.info(f"Best Hyperparameters: {best_xgb_params}")
    else:
        logger.error(f"Best hyperparameters file not found at {BEST_XGB_PARAMS_PATH}.")
        logger.error("Please ensure 006_Hyperparameter_Tuning_XGB.py ran successfully and saved the params.")
        # Fallback to parameters from 006 output if file not found (adjust if needed)
        # This is just a placeholder, it's better if the file exists.
        logger.warning("Using hardcoded fallback parameters for XGBoost. Results may not be optimal.")
        best_xgb_params = {
            'subsample': 0.9, 'reg_lambda': 1.5, 'reg_alpha': 0.05,
            'n_estimators': 400, 'min_child_weight': 5, 'max_depth': 7,
            'learning_rate': 0.01, 'gamma': 0.3, 'colsample_bytree': 0.8
        }


    logger.info(f"Loaded X_train_processed: {X_train_processed.shape}")
    logger.info(f"Loaded y_train: {y_train.shape}")

except FileNotFoundError as e:
    logger.error(f"Error: A required data/parameter file was not found: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"An error occurred while loading data/parameters: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Define SMOTE and XGBoost Model within a Pipeline
# ---------------------------------------------------------------------------
logger.info("=== 2. Defining SMOTE-XGBoost Pipeline ===")

# SMOTE - `random_state` for reproducibility.
# Corrected: Removed n_jobs as it's not supported in all versions
smote = SMOTE(random_state=42)

# XGBoost model with best parameters from tuning
# Intentionally omitting scale_pos_weight when using SMOTE to avoid over-correction.
xgb_model_smote = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False, # Suppress warning
    random_state=42,
    n_jobs=-1, # n_jobs for XGBoost itself is fine
    # Unpack parameters from the loaded dictionary
    subsample=best_xgb_params.get('subsample'),
    reg_lambda=best_xgb_params.get('reg_lambda'),
    reg_alpha=best_xgb_params.get('reg_alpha'),
    n_estimators=best_xgb_params.get('n_estimators'),
    min_child_weight=best_xgb_params.get('min_child_weight'),
    max_depth=best_xgb_params.get('max_depth'),
    learning_rate=best_xgb_params.get('learning_rate'),
    gamma=best_xgb_params.get('gamma'),
    colsample_bytree=best_xgb_params.get('colsample_bytree')
)
logger.info("XGBoost model configured with best parameters (scale_pos_weight omitted for SMOTE).")


# Create the pipeline
smote_pipeline = ImbPipeline([
    ('smote', smote),
    ('xgb', xgb_model_smote)
])

logger.info("SMOTE-XGBoost pipeline defined.")

# ---------------------------------------------------------------------------
# 3. Train the SMOTE-XGBoost Pipeline
# ---------------------------------------------------------------------------
logger.info("=== 3. Training the SMOTE-XGBoost Pipeline ===")
logger.info("This might take a moment as SMOTE is applied before training...")
start_time = time.time()
try:
    smote_pipeline.fit(X_train_processed, y_train)
except Exception as e:
    logger.error(f"Error during SMOTE-XGBoost pipeline training: {e}")
    sys.exit(1)
end_time = time.time()
logger.info(f"SMOTE-XGBoost pipeline training completed in {(end_time - start_time)/60:.2f} minutes.")

# ---------------------------------------------------------------------------
# 4. Evaluate the SMOTE-XGBoost Model
# ---------------------------------------------------------------------------
logger.info("=== 4. Evaluating the SMOTE-XGBoost Model ===")
logger.info("Making predictions on the test set with the SMOTE-XGBoost model...")
try:
    y_pred_smote = smote_pipeline.predict(X_test_processed)
    y_pred_proba_smote = smote_pipeline.predict_proba(X_test_processed)[:, 1]
except Exception as e:
    logger.error(f"Error during prediction with SMOTE-XGBoost model: {e}")
    sys.exit(1)

logger.info("\n--- SMOTE-XGBoost Model Evaluation Metrics ---")
accuracy_smote = accuracy_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote)
recall_smote = recall_score(y_test, y_pred_smote)
f1_smote = f1_score(y_test, y_pred_smote)
roc_auc_smote = roc_auc_score(y_test, y_pred_proba_smote)
pr_auc_smote = average_precision_score(y_test, y_pred_proba_smote)

logger.info(f"Accuracy:  {accuracy_smote:.4f}")
logger.info(f"Precision: {precision_smote:.4f}")
logger.info(f"Recall:    {recall_smote:.4f}")
logger.info(f"F1-score:  {f1_smote:.4f}")
logger.info(f"ROC AUC:   {roc_auc_smote:.4f}")
logger.info(f"PR AUC:    {pr_auc_smote:.4f}")

logger.info("\nSMOTE-XGBoost Model Classification Report:")
try:
    report_smote = classification_report(y_test, y_pred_smote, target_names=['Not Canceled (0)', 'Canceled (1)'])
    logger.info(f"\n{report_smote}")
except Exception as e:
    logger.error(f"Error generating classification report for SMOTE-XGBoost model: {e}")

# --- SMOTE-XGBoost Model Confusion Matrix ---
logger.info("\n--- SMOTE-XGBoost Model Confusion Matrix ---")
cm_smote = confusion_matrix(y_test, y_pred_smote)
logger.info(f"SMOTE-XGBoost Confusion Matrix:\n{cm_smote}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Predicted Not Canceled', 'Predicted Canceled'],
            yticklabels=['Actual Not Canceled', 'Actual Canceled'])
plt.title('Confusion Matrix - SMOTE-XGBoost')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
cm_smote_plot_path = PLOT_DIR / "007_confusion_matrix_xgb_smote.png"
try:
    plt.savefig(cm_smote_plot_path)
    logger.info(f"SMOTE-XGBoost confusion matrix plot saved to: {cm_smote_plot_path}")
    plt.close()
except Exception as e:
    logger.error(f"Error saving SMOTE-XGBoost confusion matrix plot: {e}")

# Feature importance for a model in a pipeline
logger.info("\n=== 4b. SMOTE-XGBoost Model Feature Importance ===")
try:
    xgb_model_from_pipeline = smote_pipeline.named_steps['xgb']
    if hasattr(xgb_model_from_pipeline, 'feature_importances_'):
        importances_smote = xgb_model_from_pipeline.feature_importances_
        feature_importance_df_smote = pd.DataFrame({
            'feature': processed_feature_names,
            'importance': importances_smote
        }).sort_values(by='importance', ascending=False)

        logger.info("\nTop 20 Features by Importance (SMOTE-XGBoost Model):")
        logger.info(feature_importance_df_smote.head(20).to_string(index=False))

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df_smote.head(20), palette='crest')
        plt.title('Top 20 Feature Importances - SMOTE-XGBoost')
        plt.tight_layout()
        fi_smote_plot_path = PLOT_DIR / "007_feature_importance_xgb_smote.png"
        plt.savefig(fi_smote_plot_path)
        logger.info(f"SMOTE-XGBoost feature importance plot saved to: {fi_smote_plot_path}")
        plt.close()
    else:
        logger.warning("The XGBoost model within the SMOTE pipeline does not have 'feature_importances_'.")
except Exception as e:
    logger.error(f"Error calculating/plotting SMOTE-XGBoost feature importance: {e}")


# ---------------------------------------------------------------------------
# 5. Save SMOTE-XGBoost Model and Results
# ---------------------------------------------------------------------------
logger.info("=== 5. Saving SMOTE-XGBoost Model and Results ===")
SMOTE_MODEL_PATH = MODEL_DIR / "xgboost_model_smote.joblib" # Defined in config.py
SMOTE_EVALUATION_METRICS_PATH = RESULTS_DIR / "007_evaluation_metrics_xgb_smote.txt"

try:
    joblib.dump(smote_pipeline, SMOTE_MODEL_PATH)
    logger.info(f"SMOTE-XGBoost pipeline model saved to: {SMOTE_MODEL_PATH}")

    with open(SMOTE_EVALUATION_METRICS_PATH, 'w') as f:
        f.write("--- SMOTE-XGBoost Model Evaluation Metrics ---\n")
        f.write(f"Used Best Hyperparameters from 006 (scale_pos_weight omitted for SMOTE):\n")
        # Log the parameters used for the XGBoost part of the pipeline
        xgb_params_in_pipeline = xgb_model_smote.get_params()
        for param, value in xgb_params_in_pipeline.items():
            # Log only parameters that were likely part of the tuning grid or core XGBoost params
            if param in best_xgb_params or param in ['objective', 'eval_metric', 'use_label_encoder', 'random_state', 'n_jobs']:
                 f.write(f"  {param}: {value}\n")
        f.write("\n--- Evaluation on Test Set ---\n")
        f.write(f"Accuracy:  {accuracy_smote:.4f}\n")
        f.write(f"Precision: {precision_smote:.4f}\n")
        f.write(f"Recall:    {recall_smote:.4f}\n")
        f.write(f"F1-score:  {f1_smote:.4f}\n")
        f.write(f"ROC AUC:   {roc_auc_smote:.4f}\n")
        f.write(f"PR AUC:    {pr_auc_smote:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_smote)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm_smote, separator=', '))
    logger.info(f"SMOTE-XGBoost evaluation metrics saved to: {SMOTE_EVALUATION_METRICS_PATH}")

except Exception as e:
    logger.error(f"Error saving SMOTE-XGBoost model or results: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 6. Comparison with Previous Tuned Model (from 006)
# ---------------------------------------------------------------------------
logger.info("=== 6. Comparison with Tuned Model (from 006) ===")
logger.info("Metrics for SMOTE-XGBoost Model:")
logger.info(f"  Precision: {precision_smote:.4f}, Recall: {recall_smote:.4f}, F1-score: {f1_smote:.4f}, PR AUC: {pr_auc_smote:.4f}")
logger.info("Compare these to the metrics from the tuned model in 006 (check 006_evaluation_metrics_xgb_tuned.txt).")
logger.info("Key considerations: Did Precision improve? How did Recall change? What about PR AUC?")


# --- End of Script ---
logger.info(f"\n--- XGBoost with SMOTE training and evaluation complete. Full output saved to: {log_filepath.resolve()} ---")

if __name__ == "__main__":
    pass