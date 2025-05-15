# 008_Model_LightGBM_Optuna_Tuning.py
"""
Trains a LightGBM model and tunes its hyperparameters using Optuna.
Compares performance to previous best models.
This is part of the main evaluation pipeline.
"""
import sys
import datetime
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
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
                        BEST_LGBM_PARAMS_OPTUNA_PATH, # Output: Path to save best hyperparams for LGBM
                        LGBM_OPTUNA_TUNED_MODEL_PATH) # Output: Path to save tuned LGBM model
    print("Successfully imported paths from config.py for 008")
except ImportError as e:
    print(f"CRITICAL (008): Error importing from config.py: {e}")
    # Fallback paths
    scripts_dir = Path(__file__).resolve().parent; project_root_alt = scripts_dir.parent
    RESULTS_DIR = project_root_alt / "results"; DATA_DIR = project_root_alt / "data"
    PLOT_DIR = project_root_alt / "plots"; MODEL_DIR = DATA_DIR / "models"
    print(f"Warning (008): Using fallback paths.")
    
    X_TRAIN_PROCESSED_PATH = DATA_DIR / "004_X_train_processed.joblib" # Corrected fallback
    Y_TRAIN_PATH = DATA_DIR / "004_y_train.joblib"                   # Corrected fallback
    X_TEST_PROCESSED_PATH = DATA_DIR / "004_X_test_processed.joblib" # Corrected fallback
    Y_TEST_PATH = DATA_DIR / "004_y_test.joblib"                   # Corrected fallback
    PROCESSED_FEATURE_NAMES_PATH = DATA_DIR / "004_processed_feature_names.joblib" # Corrected fallback
    
    BEST_LGBM_PARAMS_OPTUNA_PATH = RESULTS_DIR / "008_best_lgbm_params_optuna.joblib" # Fallback for this script's output
    LGBM_OPTUNA_TUNED_MODEL_PATH = MODEL_DIR / "008_lgbm_optuna_tuned_model.joblib"   # Fallback for this script's output
    
    for d_path in (DATA_DIR, PLOT_DIR, RESULTS_DIR, MODEL_DIR): d_path.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"CRITICAL (008): An unexpected error occurred during config import or path init: {e}")
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
log_filename_base = Path(__file__).stem # Should be "008_Model_LightGBM_Optuna_Tuning"
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(); logger.handlers.clear(); logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setFormatter(log_formatter); logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); logger.addHandler(console_handler)

# --- Start of Script ---
logger.info(f"--- LightGBM with Optuna Hyperparameter Tuning (Script {log_filename_base}) ---")
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
# 2. Define Optuna Objective Function
# ---------------------------------------------------------------------------
logger.info("=== 2. Defining Optuna Objective Function for LightGBM ===")
neg_count = np.sum(y_train == 0); pos_count = np.sum(y_train == 1)
lgbm_scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
logger.info(f"Calculated scale_pos_weight for LightGBM: {lgbm_scale_pos_weight:.2f}")

def objective(trial: optuna.Trial, X, y, cv_strategy):
    params = {
        'objective': 'binary', 'metric': 'average_precision', 'verbosity': -1, 'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'scale_pos_weight': lgbm_scale_pos_weight, 'random_state': 42, 'n_jobs': -1
    }
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  eval_metric='average_precision',
                  callbacks=[lgb.early_stopping(10, verbose=False)])
        preds_proba = model.predict_proba(X_val_fold)[:, 1]
        score = average_precision_score(y_val_fold, preds_proba)
        cv_scores.append(score)
    return np.mean(cv_scores)

# ---------------------------------------------------------------------------
# 3. Run Optuna Study
# ---------------------------------------------------------------------------
logger.info("=== 3. Running Optuna Study for LightGBM ===")
N_TRIALS = 50; N_SPLITS_CV = 3
cv_strategy_optuna = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
study = optuna.create_study(direction='maximize', study_name=f'{log_filename_base}_tuning')
logger.info(f"Starting Optuna study with {N_TRIALS} trials and {N_SPLITS_CV} CV splits...")
start_time_optuna = time.time()
try:
    study.optimize(lambda trial: objective(trial, X_train_processed, y_train, cv_strategy_optuna),
                   n_trials=N_TRIALS, n_jobs=1)
except Exception as e: logger.error(f"Error during Optuna study: {e}"); sys.exit(1)
end_time_optuna = time.time()
logger.info(f"Optuna study completed in {(end_time_optuna - start_time_optuna)/60:.2f} minutes.")
logger.info(f"Best Score (average_precision) from Optuna study: {study.best_value:.4f}")
logger.info("Best Hyperparameters found by Optuna:")
best_lgbm_params = study.best_params
# Add fixed params
best_lgbm_params.update({'objective': 'binary', 'metric': 'average_precision', 'verbosity': -1,
                         'boosting_type': 'gbdt', 'scale_pos_weight': lgbm_scale_pos_weight,
                         'random_state': 42, 'n_jobs': -1})
for param, value in best_lgbm_params.items(): logger.info(f"  {param}: {value}")

# ---------------------------------------------------------------------------
# 4. Train Final LightGBM Model with Best Parameters
# ---------------------------------------------------------------------------
logger.info("=== 4. Training Final LightGBM Model with Best Optuna Parameters ===")
final_lgbm_model = lgb.LGBMClassifier(**best_lgbm_params)
logger.info("Training final LightGBM model...")
start_time_final_train = time.time()
try: final_lgbm_model.fit(X_train_processed, y_train)
except Exception as e: logger.error(f"Error during final LightGBM model training: {e}"); sys.exit(1)
end_time_final_train = time.time()
logger.info(f"Final LightGBM model training completed in {(end_time_final_train - start_time_final_train):.2f} seconds.")

# ---------------------------------------------------------------------------
# 5. Evaluate Final LightGBM Model
# ---------------------------------------------------------------------------
logger.info("=== 5. Evaluating Final LightGBM Model ===")
logger.info("Making predictions on the test set...")
try:
    y_pred_lgbm = final_lgbm_model.predict(X_test_processed)
    y_pred_proba_lgbm = final_lgbm_model.predict_proba(X_test_processed)[:, 1]
except Exception as e: logger.error(f"Error during prediction: {e}"); sys.exit(1)

logger.info(f"\n--- LightGBM ({log_filename_base}) Evaluation Metrics ---")
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
precision_lgbm = precision_score(y_test, y_pred_lgbm, zero_division=0)
recall_lgbm = recall_score(y_test, y_pred_lgbm, zero_division=0)
f1_lgbm = f1_score(y_test, y_pred_lgbm, zero_division=0)
roc_auc_lgbm = roc_auc_score(y_test, y_pred_proba_lgbm)
pr_auc_lgbm = average_precision_score(y_test, y_pred_proba_lgbm)
logger.info(f"Accuracy:  {accuracy_lgbm:.4f}\nPrecision: {precision_lgbm:.4f}\nRecall:    {recall_lgbm:.4f}\nF1-score:  {f1_lgbm:.4f}\nROC AUC:   {roc_auc_lgbm:.4f}\nPR AUC:    {pr_auc_lgbm:.4f}")
report_lgbm = classification_report(y_test, y_pred_lgbm, target_names=['Not Canceled (0)', 'Canceled (1)'], zero_division=0)
logger.info(f"\nClassification Report:\n{report_lgbm}")
cm_lgbm = confusion_matrix(y_test, y_pred_lgbm); logger.info(f"Confusion Matrix:\n{cm_lgbm}")
plt.figure(figsize=(8, 6)); sns.heatmap(cm_lgbm, annot=True, fmt='d', cmap='coolwarm', xticklabels=['Predicted NC', 'Predicted C'], yticklabels=['Actual NC', 'Actual C'])
plt.title(f'CM - {log_filename_base}'); plt.ylabel('Actual'); plt.xlabel('Predicted')
cm_lgbm_plot_path = PLOT_DIR / f"{log_filename_base}_CM.png"
try: plt.savefig(cm_lgbm_plot_path); logger.info(f"CM plot saved to: {cm_lgbm_plot_path}"); plt.close()
except Exception as e: logger.error(f"Error saving CM plot: {e}")

logger.info(f"\n=== 5b. LightGBM ({log_filename_base}) Feature Importance ===")
try:
    if hasattr(final_lgbm_model, 'feature_importances_') and processed_feature_names is not None and X_train_processed.shape[1] == len(processed_feature_names):
        importances_lgbm = final_lgbm_model.feature_importances_
        fi_df_lgbm = pd.DataFrame({'feature': processed_feature_names, 'importance': importances_lgbm})
        fi_df_lgbm = fi_df_lgbm.sort_values(by='importance', ascending=False).head(20)
        logger.info("\nTop 20 Features by Importance:"); logger.info(fi_df_lgbm.to_string(index=False))
        plt.figure(figsize=(10, 8)); sns.barplot(x='importance', y='feature', data=fi_df_lgbm, palette='cubehelix')
        plt.title(f'Top 20 FI - {log_filename_base}'); plt.tight_layout()
        fi_lgbm_plot_path = PLOT_DIR / f"{log_filename_base}_FI.png"
        plt.savefig(fi_lgbm_plot_path); logger.info(f"FI plot saved to: {fi_lgbm_plot_path}"); plt.close()
    else: logger.warning(f"Could not generate FI. Attr: {hasattr(final_lgbm_model, 'feature_importances_')}, Names: {processed_feature_names is not None}, Len match: {X_train_processed.shape[1] == (len(processed_feature_names) if processed_feature_names else 0)}")
except Exception as e: logger.error(f"Error with FI: {e}")

# ---------------------------------------------------------------------------
# 6. Save Model and Results
# ---------------------------------------------------------------------------
logger.info(f"=== 6. Saving LightGBM ({log_filename_base}) Model and Results ===")
LGBM_EVAL_METRICS_PATH = RESULTS_DIR / f"{log_filename_base}_evaluation_metrics.txt"
try:
    joblib.dump(final_lgbm_model, LGBM_OPTUNA_TUNED_MODEL_PATH) # Uses var from config
    logger.info(f"LightGBM model saved to: {LGBM_OPTUNA_TUNED_MODEL_PATH}")
    joblib.dump(best_lgbm_params, BEST_LGBM_PARAMS_OPTUNA_PATH) # Uses var from config
    logger.info(f"Best LightGBM hyperparameters saved to: {BEST_LGBM_PARAMS_OPTUNA_PATH}")
    with open(LGBM_EVAL_METRICS_PATH, 'w') as f:
        f.write(f"--- LightGBM ({log_filename_base}) Evaluation Metrics ---\n")
        f.write(f"Best CV Score from Optuna (average_precision): {study.best_value:.4f}\nBest Hyperparameters:\n")
        for param, value in best_lgbm_params.items(): f.write(f"  {param}: {value}\n")
        f.write("\n--- Evaluation on Test Set ---\n")
        f.write(f"Accuracy:  {accuracy_lgbm:.4f}\nPrecision: {precision_lgbm:.4f}\nRecall:    {recall_lgbm:.4f}\n")
        f.write(f"F1-score:  {f1_lgbm:.4f}\nROC AUC:   {roc_auc_lgbm:.4f}\nPR AUC:    {pr_auc_lgbm:.4f}\n\n")
        f.write("Classification Report:\n" + report_lgbm + "\n\nCM:\n" + np.array2string(cm_lgbm, separator=', '))
    logger.info(f"LightGBM evaluation metrics saved to: {LGBM_EVAL_METRICS_PATH}")
except Exception as e: logger.error(f"Error saving LightGBM model or results: {e}"); sys.exit(1)

# ---------------------------------------------------------------------------
# 7. Comparison
# ---------------------------------------------------------------------------
logger.info("\n=== 7. Comparison with Previous Best Model (Tuned XGBoost) ===")
logger.info(f"Metrics for LightGBM ({log_filename_base}):")
logger.info(f"  Precision: {precision_lgbm:.4f}, Recall: {recall_lgbm:.4f}, F1-score: {f1_lgbm:.4f}, PR AUC: {pr_auc_lgbm:.4f}")
logger.info("Compare with Tuned XGBoost with Optimal Threshold (from 007_Model_XGBoost_Threshold_Tuning).")

logger.info(f"\n--- {log_filename_base} complete. Full output saved to: {log_filepath.resolve()} ---")

if __name__ == "__main__":
    pass