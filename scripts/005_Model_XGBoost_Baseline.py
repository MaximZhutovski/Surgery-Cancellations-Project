# 005_Model_XGBoost_Baseline.py
"""
Loads the preprocessed data (from 004_v3), trains an untuned XGBoost baseline model,
evaluates its performance, and saves the model and evaluation metrics.
This script uses the full engineered feature set.
"""
import sys
import datetime
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)

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
                        XGB_BASELINE_MODEL_PATH) # Corrected variable name from config
    print("Successfully imported paths from config.py for 005")
except ImportError as e:
    print(f"CRITICAL (005): Error importing from config.py: {e}")
    # Fallback paths - ensure these match the output of 004
    scripts_dir = Path(__file__).resolve().parent; project_root_alt = scripts_dir.parent
    RESULTS_DIR = project_root_alt / "results"; DATA_DIR = project_root_alt / "data"
    PLOT_DIR = project_root_alt / "plots"; MODEL_DIR = DATA_DIR / "models"
    print(f"Warning (005): Using fallback paths.")
    
    # Fallback paths for processed data - MUST MATCH 004 OUTPUT NAMES
    X_TRAIN_PROCESSED_PATH = DATA_DIR / "004_X_train_processed.joblib"
    Y_TRAIN_PATH = DATA_DIR / "004_y_train.joblib"
    X_TEST_PROCESSED_PATH = DATA_DIR / "004_X_test_processed.joblib"
    Y_TEST_PATH = DATA_DIR / "004_y_test.joblib"
    PROCESSED_FEATURE_NAMES_PATH = DATA_DIR / "004_processed_feature_names.joblib"
    # Fallback for the model this script saves
    XGB_BASELINE_MODEL_PATH = MODEL_DIR / "005_xgb_baseline_model.joblib" 
    
    for d_path in (DATA_DIR, PLOT_DIR, RESULTS_DIR, MODEL_DIR): d_path.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"CRITICAL (005): An unexpected error occurred during config import or path init: {e}")
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
log_filename_base = Path(__file__).stem # Will be "005_Model_XGBoost_Baseline"
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(); logger.handlers.clear(); logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setFormatter(log_formatter); logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); logger.addHandler(console_handler)

# --- Start of Script ---
logger.info(f"--- XGBoost Baseline Model Training (Script {log_filename_base}) ---")
logger.info(f"Using processed data from outputs of script 004 (e.g., {X_TRAIN_PROCESSED_PATH}).") # Log the path being used

# ---------------------------------------------------------------------------
# 1. Load Processed Data
# ---------------------------------------------------------------------------
logger.info("\n=== 1. Loading Processed Data ===")
try:
    X_train_processed = joblib.load(X_TRAIN_PROCESSED_PATH)
    y_train = joblib.load(Y_TRAIN_PATH)
    X_test_processed = joblib.load(X_TEST_PROCESSED_PATH)
    y_test = joblib.load(Y_TEST_PATH)
    processed_feature_names = joblib.load(PROCESSED_FEATURE_NAMES_PATH)

    logger.info(f"Loaded X_train_processed: {X_train_processed.shape}")
    logger.info(f"Loaded y_train: {y_train.shape}")
    logger.info(f"Loaded X_test_processed: {X_test_processed.shape}")
    logger.info(f"Loaded y_test: {y_test.shape}")
    if processed_feature_names is not None: # Check if it's None, not just if it's an empty list
        logger.info(f"Loaded {len(processed_feature_names)} processed feature names.")
    else:
        logger.warning("Processed feature names list is None. Feature importances might be affected.")
except FileNotFoundError as e:
    logger.error(f"Error: A required processed data file was not found: {e}")
    logger.error("Please ensure 004_Data_Final_Preprocessing.py has been run successfully and config.py is correct.")
    sys.exit(1)
except Exception as e:
    logger.error(f"An error occurred while loading processed data: {e}")
    sys.exit(1)

# (שאר הקוד של 005 נשאר זהה לגרסה הקודמת והטובה שלו,
#  הוא ישתמש ב-XGB_BASELINE_MODEL_PATH לשמירת המודל
#  וישתמש ב-log_filename_base ליצירת שמות ייחודיים לקבצי הפלט שלו)
# ...
# --- Model Training (XGBoost - Untuned) ---
logger.info("\n=== 2. Model Training (XGBoost Classifier - Untuned Baseline) ===")
try:
    neg_count = np.sum(y_train == 0); pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
    logger.info(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")
except Exception as e:
    logger.error(f"Error calculating scale_pos_weight: {e}. Defaulting to 1."); scale_pos_weight = 1

model_xgb_baseline = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    scale_pos_weight=scale_pos_weight, use_label_encoder=False,
    n_estimators=200, learning_rate=0.1, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
logger.info("Training XGBoost baseline model...")
try:
    model_xgb_baseline.fit(X_train_processed, y_train)
    logger.info("XGBoost baseline model training complete.")
except Exception as e:
    logger.error(f"Error during XGBoost baseline model training: {e}"); sys.exit(1)

# --- Model Prediction and Evaluation ---
logger.info("\n=== 3. Model Prediction and Evaluation ===")
logger.info("Making predictions on the test set...")
try:
    y_pred_xgb = model_xgb_baseline.predict(X_test_processed)
    y_pred_proba_xgb = model_xgb_baseline.predict_proba(X_test_processed)[:, 1]
except Exception as e:
    logger.error(f"Error during prediction: {e}"); sys.exit(1)

logger.info("\n--- Evaluation Metrics (XGBoost Baseline) ---")
accuracy = accuracy_score(y_test, y_pred_xgb)
precision = precision_score(y_test, y_pred_xgb, zero_division=0)
recall = recall_score(y_test, y_pred_xgb, zero_division=0)
f1 = f1_score(y_test, y_pred_xgb, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba_xgb)
pr_auc = average_precision_score(y_test, y_pred_proba_xgb)
logger.info(f"Accuracy:  {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall:    {recall:.4f}\nF1-score:  {f1:.4f}\nROC AUC:   {roc_auc:.4f}\nPR AUC:    {pr_auc:.4f}")
report = classification_report(y_test, y_pred_xgb, target_names=['Not Canceled (0)', 'Canceled (1)'], zero_division=0)
logger.info(f"\nClassification Report:\n{report}")
cm = confusion_matrix(y_test, y_pred_xgb); logger.info(f"Confusion Matrix:\n{cm}")
plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted NC', 'Predicted C'], yticklabels=['Actual NC', 'Actual C'])
plt.title(f'CM - {log_filename_base}'); plt.ylabel('Actual'); plt.xlabel('Predicted')
cm_plot_path = PLOT_DIR / f"{log_filename_base}_CM.png" # Simplified name
try: plt.savefig(cm_plot_path); logger.info(f"CM plot saved to: {cm_plot_path}"); plt.close()
except Exception as e: logger.error(f"Error saving CM plot: {e}")

fpr, tpr, _ = roc_curve(y_test, y_pred_proba_xgb)
plt.figure(figsize=(8,6)); plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})'); plt.plot([0,1],[0,1],'k--'); plt.title(f'ROC - {log_filename_base}'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(); plt.grid()
roc_plot_path = PLOT_DIR / f"{log_filename_base}_ROC.png"
try: plt.savefig(roc_plot_path); logger.info(f"ROC plot saved to: {roc_plot_path}"); plt.close()
except Exception as e: logger.error(f"Error saving ROC plot: {e}")

precision_c, recall_c, _ = precision_recall_curve(y_test, y_pred_proba_xgb)
plt.figure(figsize=(8,6)); plt.plot(recall_c, precision_c, label=f'PR (AUC = {pr_auc:.2f})'); plt.title(f'PR Curve - {log_filename_base}'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.grid()
pr_plot_path = PLOT_DIR / f"{log_filename_base}_PRC.png"
try: plt.savefig(pr_plot_path); logger.info(f"PR curve plot saved to: {pr_plot_path}"); plt.close()
except Exception as e: logger.error(f"Error saving PR plot: {e}")

# --- Feature Importance ---
logger.info("\n=== 4. Feature Importance (XGBoost Baseline) ===")
try:
    if hasattr(model_xgb_baseline, 'feature_importances_') and processed_feature_names is not None and X_train_processed.shape[1] == len(processed_feature_names) :
        importances = model_xgb_baseline.feature_importances_
        fi_df = pd.DataFrame({'feature': processed_feature_names, 'importance': importances})
        fi_df = fi_df.sort_values(by='importance', ascending=False).head(20)
        logger.info("\nTop 20 Features by Importance:"); logger.info(fi_df.to_string(index=False))
        plt.figure(figsize=(10, 8)); sns.barplot(x='importance', y='feature', data=fi_df, palette='viridis')
        plt.title(f'Top 20 FI - {log_filename_base}'); plt.tight_layout()
        fi_plot_path = PLOT_DIR / f"{log_filename_base}_FI.png"
        plt.savefig(fi_plot_path); logger.info(f"FI plot saved to: {fi_plot_path}"); plt.close()
    else: logger.warning(f"Could not generate FI. Attr: {hasattr(model_xgb_baseline, 'feature_importances_')}, Names: {processed_feature_names is not None}, Len match: {X_train_processed.shape[1] == (len(processed_feature_names) if processed_feature_names else 0)}")
except Exception as e: logger.error(f"Error with FI: {e}")

# --- Save Model and Metrics ---
logger.info("\n=== 5. Saving Trained Model and Metrics ===")
EVALUATION_METRICS_PATH = RESULTS_DIR / f"{log_filename_base}_evaluation_metrics.txt"
try:
    joblib.dump(model_xgb_baseline, XGB_BASELINE_MODEL_PATH) # Uses var from config
    logger.info(f"Trained XGBoost baseline model saved to: {XGB_BASELINE_MODEL_PATH}")
    with open(EVALUATION_METRICS_PATH, 'w') as f:
        f.write(f"--- XGBoost Baseline Model ({log_filename_base}) ---\n")
        f.write(f"Accuracy:  {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall:    {recall:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\nROC AUC:   {roc_auc:.4f}\nPR AUC:    {pr_auc:.4f}\n\n")
        f.write("Classification Report:\n" + report + "\n\nCM:\n" + np.array2string(cm, separator=', '))
    logger.info(f"Evaluation metrics saved to: {EVALUATION_METRICS_PATH}")
except Exception as e: logger.error(f"Error saving model/metrics: {e}"); sys.exit(1)

logger.info("\n=== 6. Summary and Next Steps ===") # ... (summary)
logger.info(f"XGBoost baseline model ({log_filename_base}) training and evaluation complete.")
logger.info(f"Key metrics: Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1:.4f}, PR AUC: {pr_auc:.4f}")
logger.info(f"Next step is 006_Model_XGBoost_Tuning.py to tune hyperparameters for this feature set.")
logger.info(f"\n--- {log_filename_base} complete. Full output saved to: {log_filepath.resolve()} ---")

if __name__ == "__main__":
    pass