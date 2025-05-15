# 006_Model_XGBoost_Tuning.py
"""
Performs hyperparameter tuning for the XGBoost model using RandomizedSearchCV
on the full engineered feature set from 004.
Trains a new model with the best found parameters and evaluates it.
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
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
                        X_TRAIN_PROCESSED_PATH, Y_TRAIN_PATH,
                        X_TEST_PROCESSED_PATH, Y_TEST_PATH,
                        PROCESSED_FEATURE_NAMES_PATH,
                        BEST_XGB_PARAMS_PATH,  # Output: Path to save best hyperparams
                        XGB_TUNED_MODEL_PATH)  # Output: Path to save tuned model
    print("Successfully imported paths from config.py for 006")
except ImportError as e:
    print(f"CRITICAL (006): Error importing from config.py: {e}")
    # Fallback paths
    scripts_dir = Path(__file__).resolve().parent; project_root_alt = scripts_dir.parent
    RESULTS_DIR = project_root_alt / "results"; DATA_DIR = project_root_alt / "data"
    PLOT_DIR = project_root_alt / "plots"; MODEL_DIR = DATA_DIR / "models"
    print(f"Warning (006): Using fallback paths.")
    X_TRAIN_PROCESSED_PATH = DATA_DIR / "004_X_train_processed.joblib"
    Y_TRAIN_PATH = DATA_DIR / "004_y_train.joblib"
    X_TEST_PROCESSED_PATH = DATA_DIR / "004_X_test_processed.joblib"
    Y_TEST_PATH = DATA_DIR / "004_y_test.joblib"
    PROCESSED_FEATURE_NAMES_PATH = DATA_DIR / "004_processed_feature_names.joblib"
    BEST_XGB_PARAMS_PATH = RESULTS_DIR / "006_best_xgb_params.joblib"
    XGB_TUNED_MODEL_PATH = MODEL_DIR / "006_xgb_tuned_model.joblib"
    for d_path in (DATA_DIR, PLOT_DIR, RESULTS_DIR, MODEL_DIR): d_path.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"CRITICAL (006): An unexpected error occurred during config import or path init: {e}")
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
log_filename_base = Path(__file__).stem # Should be "006_Model_XGBoost_Tuning"
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(); logger.handlers.clear(); logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setFormatter(log_formatter); logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); logger.addHandler(console_handler)

# --- Start of Script ---
logger.info(f"--- XGBoost Hyperparameter Tuning (Script {log_filename_base}) ---")
logger.info(f"Using processed data from outputs of script 004 (e.g., {X_TRAIN_PROCESSED_PATH}).")

# ---------------------------------------------------------------------------
# 1. Load Processed Data
# ---------------------------------------------------------------------------
logger.info("=== 1. Loading Processed Data ===")
try:
    X_train_processed = joblib.load(X_TRAIN_PROCESSED_PATH)
    y_train = joblib.load(Y_TRAIN_PATH)
    X_test_processed = joblib.load(X_TEST_PROCESSED_PATH)
    y_test = joblib.load(Y_TEST_PATH)
    processed_feature_names = joblib.load(PROCESSED_FEATURE_NAMES_PATH)
    logger.info(f"Loaded X_train_processed: {X_train_processed.shape}")
    if processed_feature_names is not None: logger.info(f"Loaded {len(processed_feature_names)} feature names.")
    else: logger.warning("Processed feature names list is None.")
except FileNotFoundError as e:
    logger.error(f"Error: A required data file was not found: {e}"); sys.exit(1)
except Exception as e:
    logger.error(f"An error occurred while loading data: {e}"); sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Define Hyperparameter Grid and Perform Randomized Search
# ---------------------------------------------------------------------------
logger.info("=== 2. Hyperparameter Tuning with RandomizedSearchCV ===")
neg_count = np.sum(y_train == 0); pos_count = np.sum(y_train == 1)
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500, 600],
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'max_depth': [4, 5, 6, 7, 8, 9, 10],
    'min_child_weight': [1, 3, 5, 7, 9],
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1],
    'reg_lambda': [0.1, 0.5, 1, 1.5, 2, 3, 5]
}
xgb_estimator = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                                  scale_pos_weight=scale_pos_weight, use_label_encoder=False,
                                  random_state=42, n_jobs=-1)
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Reduced to 3 for speed, can be 5
N_ITER_RANDOM_SEARCH = 60 # Number of iterations for RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_estimator, param_distributions=param_grid,
    n_iter=N_ITER_RANDOM_SEARCH,
    scoring='average_precision', cv=cv_strategy,
    verbose=1, random_state=42, n_jobs=-1
)

logger.info(f"Starting RandomizedSearchCV (n_iter={random_search.n_iter}, cv_splits={cv_strategy.get_n_splits()})...")
start_time = time.time()
try: random_search.fit(X_train_processed, y_train)
except Exception as e: logger.error(f"Error during RandomizedSearchCV: {e}"); sys.exit(1)
end_time = time.time()
logger.info(f"RandomizedSearchCV completed in {(end_time - start_time)/60:.2f} minutes.")
logger.info(f"Best Score (average_precision) from RandomizedSearchCV: {random_search.best_score_:.4f}")
logger.info("Best Hyperparameters found:")
best_params_dict = random_search.best_params_
for param, value in best_params_dict.items(): logger.info(f"  {param}: {value}")

# ---------------------------------------------------------------------------
# 3. Train Final Model with Best Parameters
# ---------------------------------------------------------------------------
logger.info("=== 3. Training Final Model with Best Parameters ===")
xgb_tuned_model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss',
    scale_pos_weight=scale_pos_weight, use_label_encoder=False,
    random_state=42, n_jobs=-1, **best_params_dict
)
logger.info("Training final XGBoost model with best parameters...")
try: xgb_tuned_model.fit(X_train_processed, y_train)
except Exception as e: logger.error(f"Error during final model training: {e}"); sys.exit(1)
logger.info("Final XGBoost model training complete.")

# ---------------------------------------------------------------------------
# 4. Evaluate Tuned Model
# ---------------------------------------------------------------------------
logger.info("\n=== 4. Evaluating Tuned XGBoost Model ===")
logger.info("Making predictions on the test set with the tuned model...")
try:
    y_pred_tuned = xgb_tuned_model.predict(X_test_processed)
    y_pred_proba_tuned = xgb_tuned_model.predict_proba(X_test_processed)[:, 1]
except Exception as e: logger.error(f"Error during prediction: {e}"); sys.exit(1)

logger.info("\n--- Tuned XGBoost Model Evaluation Metrics ---")
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned, zero_division=0)
recall_tuned = recall_score(y_test, y_pred_tuned, zero_division=0)
f1_tuned = f1_score(y_test, y_pred_tuned, zero_division=0)
roc_auc_tuned = roc_auc_score(y_test, y_pred_proba_tuned)
pr_auc_tuned = average_precision_score(y_test, y_pred_proba_tuned)

logger.info(f"Accuracy:  {accuracy_tuned:.4f}")
logger.info(f"Precision: {precision_tuned:.4f}")
logger.info(f"Recall:    {recall_tuned:.4f}")
logger.info(f"F1-score:  {f1_tuned:.4f}")
logger.info(f"ROC AUC:   {roc_auc_tuned:.4f}")
logger.info(f"PR AUC:    {pr_auc_tuned:.4f}")

report_tuned = classification_report(y_test, y_pred_tuned, target_names=['Not Canceled (0)', 'Canceled (1)'], zero_division=0)
logger.info(f"\nTuned Model Classification Report:\n{report_tuned}")
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
logger.info(f"Tuned Model Confusion Matrix:\n{cm_tuned}")

plt.figure(figsize=(8, 6)); sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Greens', xticklabels=['Predicted NC', 'Predicted C'], yticklabels=['Actual NC', 'Actual C'])
plt.title(f'CM - {log_filename_base}'); plt.ylabel('Actual'); plt.xlabel('Predicted')
cm_tuned_plot_path = PLOT_DIR / f"{log_filename_base}_CM.png"
try: plt.savefig(cm_tuned_plot_path); logger.info(f"CM plot saved to: {cm_tuned_plot_path}"); plt.close()
except Exception as e: logger.error(f"Error saving CM plot: {e}")

# Feature Importance
logger.info("\n=== 4b. Tuned XGBoost Model Feature Importance ===")
try:
    if hasattr(xgb_tuned_model, 'feature_importances_') and processed_feature_names is not None and X_train_processed.shape[1] == len(processed_feature_names):
        importances_tuned = xgb_tuned_model.feature_importances_
        fi_df_tuned = pd.DataFrame({'feature': processed_feature_names, 'importance': importances_tuned})
        fi_df_tuned = fi_df_tuned.sort_values(by='importance', ascending=False).head(20)
        logger.info("\nTop 20 Features by Importance (Tuned Model):"); logger.info(fi_df_tuned.to_string(index=False))
        plt.figure(figsize=(10, 8)); sns.barplot(x='importance', y='feature', data=fi_df_tuned, palette='mako')
        plt.title(f'Top 20 FI - {log_filename_base}'); plt.tight_layout()
        fi_tuned_plot_path = PLOT_DIR / f"{log_filename_base}_FI.png"
        plt.savefig(fi_tuned_plot_path); logger.info(f"FI plot saved to: {fi_tuned_plot_path}"); plt.close()
    else: logger.warning(f"Could not generate FI. Attr: {hasattr(xgb_tuned_model, 'feature_importances_')}, Names: {processed_feature_names is not None}, Len match: {X_train_processed.shape[1] == (len(processed_feature_names) if processed_feature_names else 0)}")
except Exception as e: logger.error(f"Error with FI: {e}")

# ---------------------------------------------------------------------------
# 5. Save Tuned Model and Results
# ---------------------------------------------------------------------------
logger.info("=== 5. Saving Tuned Model and Results ===")
TUNED_EVAL_METRICS_PATH = RESULTS_DIR / f"{log_filename_base}_evaluation_metrics.txt"
try:
    joblib.dump(xgb_tuned_model, XGB_TUNED_MODEL_PATH) # Uses variable from config
    logger.info(f"Tuned XGBoost model saved to: {XGB_TUNED_MODEL_PATH}")
    joblib.dump(best_params_dict, BEST_XGB_PARAMS_PATH) # Uses variable from config
    logger.info(f"Best hyperparameters saved to: {BEST_XGB_PARAMS_PATH}")
    with open(TUNED_EVAL_METRICS_PATH, 'w') as f:
        f.write(f"--- Tuned XGBoost Model Evaluation Metrics ({log_filename_base}) ---\n")
        f.write(f"Best CV Score (average_precision): {random_search.best_score_:.4f}\nBest Hyperparameters:\n")
        for param, value in best_params_dict.items(): f.write(f"  {param}: {value}\n")
        f.write("\n--- Evaluation on Test Set ---\n")
        f.write(f"Accuracy:  {accuracy_tuned:.4f}\nPrecision: {precision_tuned:.4f}\nRecall:    {recall_tuned:.4f}\n")
        f.write(f"F1-score:  {f1_tuned:.4f}\nROC AUC:   {roc_auc_tuned:.4f}\nPR AUC:    {pr_auc_tuned:.4f}\n\n")
        f.write("Classification Report:\n" + report_tuned + "\n\nCM:\n" + np.array2string(cm_tuned, separator=', '))
    logger.info(f"Tuned evaluation metrics saved to: {TUNED_EVAL_METRICS_PATH}")
except Exception as e: logger.error(f"Error saving model or results: {e}"); sys.exit(1)

# ---------------------------------------------------------------------------
# 6. Comparison with Previous Models
# ---------------------------------------------------------------------------
logger.info("\n=== 6. Comparison with Baseline Model ===")
logger.info(f"Results for Tuned XGBoost ({log_filename_base}):")
logger.info(f"  Precision: {precision_tuned:.4f}, Recall: {recall_tuned:.4f}, F1-score: {f1_tuned:.4f}, PR AUC: {pr_auc_tuned:.4f}")
logger.info(f"Compare with XGBoost Baseline (from 005_Model_XGBoost_Baseline).")
logger.info(f"Next step is 007_Model_XGBoost_Threshold_Tuning.py")

logger.info(f"\n--- {log_filename_base} complete. Full output saved to: {log_filepath.resolve()} ---")

if __name__ == "__main__":
    pass