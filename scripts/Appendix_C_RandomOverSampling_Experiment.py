# Appendix_C_RandomOverSampling_Experiment.py
"""
Trains an XGBoost model using RandomOverSampler for handling class imbalance.
Uses best hyperparameters from the main XGBoost tuning script (006).
Compares results with previous models.
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
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score)
import matplotlib
matplotlib.use('Agg')
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
                        X_TRAIN_PROCESSED_PATH, Y_TRAIN_PATH,           # Input from 004
                        X_TEST_PROCESSED_PATH, Y_TEST_PATH,            # Input from 004
                        PROCESSED_FEATURE_NAMES_PATH,                  # Input from 004
                        BEST_XGB_PARAMS_PATH,       # Input: Best params from 006
                        # Define new model path for this script's output (as an appendix model)
                        XGB_MODEL_ROS_PATH) # Optional: if you want to save this specific model via config
    print(f"Successfully imported paths from config.py for {Path(__file__).stem}")
except ImportError as e:
    print(f"CRITICAL ({Path(__file__).stem}): Error importing from config.py: {e}")
    # Fallback paths
    scripts_dir = Path(__file__).resolve().parent; project_root_alt = scripts_dir.parent
    RESULTS_DIR = project_root_alt / "results"; DATA_DIR = project_root_alt / "data"
    PLOT_DIR = project_root_alt / "plots"; MODEL_DIR = DATA_DIR / "models"
    print(f"Warning ({Path(__file__).stem}): Using fallback paths.")
    
    X_TRAIN_PROCESSED_PATH = DATA_DIR / "004_X_train_processed.joblib" # Corrected fallback
    Y_TRAIN_PATH = DATA_DIR / "004_y_train.joblib"                   # Corrected fallback
    X_TEST_PROCESSED_PATH = DATA_DIR / "004_X_test_processed.joblib" # Corrected fallback
    Y_TEST_PATH = DATA_DIR / "004_y_test.joblib"                   # Corrected fallback
    PROCESSED_FEATURE_NAMES_PATH = DATA_DIR / "004_processed_feature_names.joblib" # Corrected fallback
    
    BEST_XGB_PARAMS_PATH = RESULTS_DIR / "006_best_xgb_params.joblib" # Fallback for params from main 006
    # Fallback for the model this script saves (if not defined in config)
    if 'XGB_MODEL_ROS_PATH' not in globals():
        XGB_MODEL_ROS_PATH = MODEL_DIR / "appendix_C_xgb_ros_model.joblib"
    
    for d_path in (DATA_DIR, PLOT_DIR, RESULTS_DIR, MODEL_DIR): d_path.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"CRITICAL ({Path(__file__).stem}): An unexpected error during config import or path init: {e}")
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
log_filename_base = Path(__file__).stem # e.g., "Appendix_C_RandomOverSampling_Experiment"
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(); logger.handlers.clear(); logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setFormatter(log_formatter); logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); logger.addHandler(console_handler)

# --- Start of Script ---
logger.info(f"--- XGBoost with RandomOverSampler (Script {log_filename_base}) ---")
logger.info(f"Using processed data from outputs of script 004 (e.g., {X_TRAIN_PROCESSED_PATH}).")
logger.info(f"Using best XGBoost params from: {BEST_XGB_PARAMS_PATH}")


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
        best_xgb_params_from_main_tuning = joblib.load(BEST_XGB_PARAMS_PATH)
        logger.info("Loaded best XGBoost hyperparameters from main tuning script (006).")
        logger.info(f"Best Hyperparameters: {best_xgb_params_from_main_tuning}")
    else:
        logger.error(f"Best hyperparameters file from main tuning (006) not found at {BEST_XGB_PARAMS_PATH}.")
        sys.exit(1)

    logger.info(f"Loaded X_train_processed: {X_train_processed.shape}")
    logger.info(f"Loaded y_train. Positive class proportion: {y_train.mean():.2f}")
except FileNotFoundError as e:
    logger.error(f"Error: A required data/parameter file was not found: {e}"); sys.exit(1)
except Exception as e:
    logger.error(f"An error occurred while loading data/parameters: {e}"); sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Define RandomOverSampler and XGBoost Model within a Pipeline
# ---------------------------------------------------------------------------
logger.info("=== 2. Defining RandomOverSampler-XGBoost Pipeline ===")
ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
xgb_params_for_ros = best_xgb_params_from_main_tuning.copy()
if 'scale_pos_weight' in xgb_params_for_ros:
    logger.info(f"Original scale_pos_weight was {xgb_params_for_ros['scale_pos_weight']}. Setting to 1 for RandomOverSampler.")
    xgb_params_for_ros['scale_pos_weight'] = 1
else:
    xgb_params_for_ros['scale_pos_weight'] = 1
xgb_model_ros = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss', use_label_encoder=False,
    random_state=42, n_jobs=-1, **xgb_params_for_ros
)
logger.info("XGBoost model configured with best parameters and scale_pos_weight=1 for RandomOverSampler.")
ros_pipeline = ImbPipeline([('ros', ros), ('xgb', xgb_model_ros)])
logger.info("RandomOverSampler-XGBoost pipeline defined.")

# ---------------------------------------------------------------------------
# 3. Train the Pipeline
# ---------------------------------------------------------------------------
logger.info("=== 3. Training the RandomOverSampler-XGBoost Pipeline ===")
start_time = time.time()
try: ros_pipeline.fit(X_train_processed, y_train)
except Exception as e: logger.error(f"Error during pipeline training: {e}"); sys.exit(1)
end_time = time.time()
logger.info(f"Pipeline training completed in {(end_time - start_time)/60:.2f} minutes.")

# ---------------------------------------------------------------------------
# 4. Evaluate the Model
# ---------------------------------------------------------------------------
logger.info("=== 4. Evaluating the RandomOverSampler-XGBoost Model ===")
logger.info("Making predictions on the test set...")
try:
    y_pred_ros = ros_pipeline.predict(X_test_processed)
    y_pred_proba_ros = ros_pipeline.predict_proba(X_test_processed)[:, 1]
except Exception as e: logger.error(f"Error during prediction: {e}"); sys.exit(1)

logger.info(f"\n--- RandomOverSampler-XGBoost Model ({log_filename_base}) Evaluation Metrics ---")
accuracy_ros = accuracy_score(y_test, y_pred_ros)
precision_ros = precision_score(y_test, y_pred_ros, zero_division=0)
recall_ros = recall_score(y_test, y_pred_ros, zero_division=0)
f1_ros = f1_score(y_test, y_pred_ros, zero_division=0)
roc_auc_ros = roc_auc_score(y_test, y_pred_proba_ros)
pr_auc_ros = average_precision_score(y_test, y_pred_proba_ros)
logger.info(f"Accuracy:  {accuracy_ros:.4f}\nPrecision: {precision_ros:.4f}\nRecall:    {recall_ros:.4f}\nF1-score:  {f1_ros:.4f}\nROC AUC:   {roc_auc_ros:.4f}\nPR AUC:    {pr_auc_ros:.4f}")
report_ros = classification_report(y_test, y_pred_ros, target_names=['Not Canceled (0)', 'Canceled (1)'], zero_division=0)
logger.info(f"\nClassification Report:\n{report_ros}")
cm_ros = confusion_matrix(y_test, y_pred_ros); logger.info(f"Confusion Matrix:\n{cm_ros}")
plt.figure(figsize=(8, 6)); sns.heatmap(cm_ros, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Predicted NC', 'Predicted C'], yticklabels=['Actual NC', 'Actual C'])
plt.title(f'CM - {log_filename_base}'); plt.ylabel('Actual'); plt.xlabel('Predicted')
cm_ros_plot_path = PLOT_DIR / f"{log_filename_base}_CM.png"
try: plt.savefig(cm_ros_plot_path); logger.info(f"CM plot saved to: {cm_ros_plot_path}"); plt.close()
except Exception as e: logger.error(f"Error saving CM plot: {e}")

logger.info(f"\n=== 4b. {log_filename_base} Feature Importance ===")
try:
    xgb_model_from_pipeline_ros = ros_pipeline.named_steps['xgb']
    if hasattr(xgb_model_from_pipeline_ros, 'feature_importances_') and processed_feature_names is not None and X_train_processed.shape[1] == len(processed_feature_names) :
        importances_ros = xgb_model_from_pipeline_ros.feature_importances_
        fi_df_ros = pd.DataFrame({'feature': processed_feature_names, 'importance': importances_ros})
        fi_df_ros = fi_df_ros.sort_values(by='importance', ascending=False).head(20)
        logger.info("\nTop 20 Features by Importance:"); logger.info(fi_df_ros.to_string(index=False))
        plt.figure(figsize=(10, 8)); sns.barplot(x='importance', y='feature', data=fi_df_ros, palette='viridis_r')
        plt.title(f'Top 20 FI - {log_filename_base}'); plt.tight_layout()
        fi_ros_plot_path = PLOT_DIR / f"{log_filename_base}_FI.png"
        plt.savefig(fi_ros_plot_path); logger.info(f"FI plot saved to: {fi_ros_plot_path}"); plt.close()
    else: logger.warning(f"Could not generate FI. Attr: {hasattr(xgb_model_from_pipeline_ros, 'feature_importances_')}, Names: {processed_feature_names is not None}, Len match: {X_train_processed.shape[1] == (len(processed_feature_names) if processed_feature_names else 0)}")
except Exception as e: logger.error(f"Error with FI: {e}")

# ---------------------------------------------------------------------------
# 5. Save Model and Results
# ---------------------------------------------------------------------------
logger.info(f"=== 5. Saving {log_filename_base} Model and Results ===")
ROS_EVAL_METRICS_PATH = RESULTS_DIR / f"{log_filename_base}_evaluation_metrics.txt"
try:
    # If XGB_MODEL_ROS_PATH was successfully imported from config, use it. Otherwise, use a default.
    model_save_path = XGB_MODEL_ROS_PATH if 'XGB_MODEL_ROS_PATH' in globals() else MODEL_DIR / f"{log_filename_base}_model.joblib"
    joblib.dump(ros_pipeline, model_save_path)
    logger.info(f"{log_filename_base} pipeline model saved to: {model_save_path}")
    with open(ROS_EVAL_METRICS_PATH, 'w') as f:
        f.write(f"--- {log_filename_base} Evaluation Metrics ---\n")
        f.write(f"Used Best Hyperparameters from main XGBoost tuning (with scale_pos_weight=1):\n")
        for param, value in xgb_params_for_ros.items(): f.write(f"  {param}: {value}\n")
        f.write("\n--- Evaluation on Test Set ---\n")
        f.write(f"Accuracy:  {accuracy_ros:.4f}\nPrecision: {precision_ros:.4f}\nRecall:    {recall_ros:.4f}\n")
        f.write(f"F1-score:  {f1_ros:.4f}\nROC AUC:   {roc_auc_ros:.4f}\nPR AUC:    {pr_auc_ros:.4f}\n\n")
        f.write("Classification Report:\n" + report_ros + "\n\nCM:\n" + np.array2string(cm_ros, separator=', '))
    logger.info(f"{log_filename_base} evaluation metrics saved to: {ROS_EVAL_METRICS_PATH}")
except Exception as e: logger.error(f"Error saving model or results: {e}"); sys.exit(1)

# ---------------------------------------------------------------------------
# 6. Comparison
# ---------------------------------------------------------------------------
logger.info("\n=== 6. Comparison with Previous Models ===")
logger.info(f"Metrics for {log_filename_base}:")
logger.info(f"  Precision: {precision_ros:.4f}, Recall: {recall_ros:.4f}, F1-score: {f1_ros:.4f}, PR AUC: {pr_auc_ros:.4f}")
logger.info("Compare with:")
logger.info("  1. Tuned XGBoost (006_Model_XGBoost_Tuning with optimal threshold from 007).")
logger.info("  2. Tuned LightGBM (008_Model_LightGBM_Optuna_Tuning with optimal threshold from 009).")

logger.info(f"\n--- {log_filename_base} training and evaluation complete. Full output saved to: {log_filepath.resolve()} ---")

if __name__ == "__main__":
    pass