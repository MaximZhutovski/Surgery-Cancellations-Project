# 009_Model_LightGBM_Threshold_Tuning.py
"""
Performs threshold tuning for the tuned LightGBM model
(from script 008_Model_LightGBM_Optuna_Tuning.py)
to find an optimal balance between precision and recall.
"""
import sys
import datetime
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             classification_report, precision_recall_curve, average_precision_score)

# ---------------------------------------------------------------------------
# Setup: Add project root to PYTHONPATH and import config
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config import (RESULTS_DIR, DATA_DIR, PLOT_DIR, MODEL_DIR,
                        X_TEST_PROCESSED_PATH, Y_TEST_PATH,                # Input data from 004
                        LGBM_OPTUNA_TUNED_MODEL_PATH) # Input model from 008 (corrected name)
    print("Successfully imported paths from config.py for 009")
except ImportError as e:
    print(f"CRITICAL (009): Error importing from config.py: {e}")
    # Fallback paths
    scripts_dir = Path(__file__).resolve().parent; project_root_alt = scripts_dir.parent
    RESULTS_DIR = project_root_alt / "results"; DATA_DIR = project_root_alt / "data"
    PLOT_DIR = project_root_alt / "plots"; MODEL_DIR = DATA_DIR / "models"
    print(f"Warning (009): Using fallback paths.")
    
    X_TEST_PROCESSED_PATH = DATA_DIR / "004_X_test_processed.joblib" # Corrected fallback
    Y_TEST_PATH = DATA_DIR / "004_y_test.joblib"                   # Corrected fallback
    # Corrected fallback for the model this script loads
    LGBM_OPTUNA_TUNED_MODEL_PATH = MODEL_DIR / "008_lgbm_optuna_tuned_model.joblib" 
    
    for d_path in (DATA_DIR, PLOT_DIR, RESULTS_DIR, MODEL_DIR): d_path.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"CRITICAL (009): An unexpected error occurred during config import or path init: {e}")
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
log_filename_base = Path(__file__).stem # Should be "009_Model_LightGBM_Threshold_Tuning"
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(); logger.handlers.clear(); logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setFormatter(log_formatter); logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); logger.addHandler(console_handler)

# --- Start of Script ---
logger.info(f"--- Threshold Tuning for Tuned LightGBM (Script {log_filename_base}) ---")
logger.info(f"Using tuned model from: {LGBM_OPTUNA_TUNED_MODEL_PATH}")
logger.info(f"Using test data from: {X_TEST_PROCESSED_PATH}")

# ---------------------------------------------------------------------------
# 1. Load Processed Test Data and Tuned LightGBM Model
# ---------------------------------------------------------------------------
logger.info("=== 1. Loading Processed Test Data and Tuned LightGBM Model ===")
try:
    X_test_processed = joblib.load(X_TEST_PROCESSED_PATH)
    y_test = joblib.load(Y_TEST_PATH)
    
    if not LGBM_OPTUNA_TUNED_MODEL_PATH.exists():
        logger.error(f"Tuned LightGBM model file not found at: {LGBM_OPTUNA_TUNED_MODEL_PATH}")
        logger.error(f"Please ensure script 008 (e.g., 008_Model_LightGBM_Optuna_Tuning.py) ran successfully and saved the model.")
        sys.exit(1)
    tuned_lgbm_model = joblib.load(LGBM_OPTUNA_TUNED_MODEL_PATH)

    logger.info(f"Loaded X_test_processed: {X_test_processed.shape}")
    logger.info(f"Tuned LightGBM model loaded successfully from: {LGBM_OPTUNA_TUNED_MODEL_PATH}")
except FileNotFoundError as e:
    logger.error(f"Error: A required data/model file was not found: {e}"); sys.exit(1)
except Exception as e:
    logger.error(f"An error occurred while loading data/model: {e}"); sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Predict Probabilities
# ---------------------------------------------------------------------------
logger.info("=== 2. Predicting Probabilities on Test Set (Tuned LightGBM) ===")
try:
    y_pred_proba_tuned_lgbm = tuned_lgbm_model.predict_proba(X_test_processed)[:, 1]
    logger.info("Probabilities predicted successfully.")
except Exception as e:
    logger.error(f"Error during probability prediction: {e}"); sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Iterate Through Thresholds and Calculate Metrics
# ---------------------------------------------------------------------------
logger.info("=== 3. Evaluating Metrics for Different Thresholds (Tuned LightGBM) ===")
thresholds = np.arange(0.05, 0.96, 0.01)
results_lgbm = []
for thresh in thresholds:
    y_pred_thresh_lgbm = (y_pred_proba_tuned_lgbm >= thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_thresh_lgbm, pos_label=1, average='binary', zero_division=0)
    results_lgbm.append({'threshold': thresh, 'precision': precision, 'recall': recall, 'f1_score': f1})
results_df_lgbm = pd.DataFrame(results_lgbm)

logger.info("\nMetrics for various thresholds (sample - tuned LightGBM):")
logger.info(results_df_lgbm.iloc[::10].to_string(index=False))
threshold_results_csv_path_lgbm = RESULTS_DIR / f"{log_filename_base}_threshold_tuning_results.csv"
results_df_lgbm.to_csv(threshold_results_csv_path_lgbm, index=False)
logger.info(f"\nFull threshold tuning results (LightGBM) saved to: {threshold_results_csv_path_lgbm}")

# ---------------------------------------------------------------------------
# 4. Plot Precision, Recall, and F1-score vs. Threshold
# ---------------------------------------------------------------------------
logger.info("=== 4. Plotting Metrics vs. Threshold (Tuned LightGBM) ===")
plt.figure(figsize=(12, 7))
plt.plot(results_df_lgbm['threshold'], results_df_lgbm['precision'], label='Precision', color='blue', marker='.')
plt.plot(results_df_lgbm['threshold'], results_df_lgbm['recall'], label='Recall', color='green', marker='.')
plt.plot(results_df_lgbm['threshold'], results_df_lgbm['f1_score'], label='F1-score', color='red', marker='.')
plt.title(f'P, R, F1 vs. Threshold ({log_filename_base})')
plt.xlabel('Threshold'); plt.ylabel('Score'); plt.legend(); plt.grid(True); plt.xticks(np.arange(0.0, 1.01, 0.05))
plt.axvline(x=0.5, color='grey', linestyle='--', label='Default Thresh (0.5)')

best_f1_row_lgbm = results_df_lgbm.loc[results_df_lgbm['f1_score'].idxmax()] if not results_df_lgbm.empty else None
best_f1_threshold_lgbm_iter = 0.5
if best_f1_row_lgbm is not None:
    best_f1_threshold_lgbm_iter = best_f1_row_lgbm['threshold']
    max_f1_score_lgbm_iter = best_f1_row_lgbm['f1_score']
    plt.axvline(x=best_f1_threshold_lgbm_iter, color='red', linestyle=':', label=f'Max F1 Iter ({best_f1_threshold_lgbm_iter:.3f})')
    logger.info(f"\nThreshold for Max F1-score (from iteration): {best_f1_threshold_lgbm_iter:.3f} (F1: {max_f1_score_lgbm_iter:.4f})")
else:
    logger.warning("Could not determine best F1 threshold from iteration for LightGBM.")

plt.legend()
threshold_plot_path_lgbm = PLOT_DIR / f"{log_filename_base}_metrics_vs_threshold_plot.png"
try: plt.savefig(threshold_plot_path_lgbm); logger.info(f"Threshold metrics plot saved to: {threshold_plot_path_lgbm}"); plt.close()
except Exception as e: logger.error(f"Error saving threshold metrics plot: {e}")

precision_curve_lgbm, recall_curve_lgbm, pr_thresholds_lgbm = precision_recall_curve(y_test, y_pred_proba_tuned_lgbm)
avg_precision_lgbm = average_precision_score(y_test, y_pred_proba_tuned_lgbm)
plt.figure(figsize=(10, 7)); plt.plot(recall_curve_lgbm, precision_curve_lgbm, lw=2, color='purple', label=f'PR curve (AUC = {avg_precision_lgbm:.3f})')
optimal_threshold_pr_lgbm = 0.5
if len(pr_thresholds_lgbm) > 0:
    f1_scores_on_pr_curve_lgbm = (2 * precision_curve_lgbm[:-1] * recall_curve_lgbm[:-1]) / (precision_curve_lgbm[:-1] + recall_curve_lgbm[:-1] + 1e-9)
    if len(f1_scores_on_pr_curve_lgbm) > 0:
        optimal_idx_pr_lgbm = np.argmax(f1_scores_on_pr_curve_lgbm)
        if optimal_idx_pr_lgbm < len(pr_thresholds_lgbm):
            optimal_threshold_pr_lgbm = pr_thresholds_lgbm[optimal_idx_pr_lgbm]
            plt.scatter(recall_curve_lgbm[optimal_idx_pr_lgbm], precision_curve_lgbm[optimal_idx_pr_lgbm], marker='o', color='red', s=100,
                        label=f'Best F1 (Thresh≈{optimal_threshold_pr_lgbm:.3f})')
            logger.info(f"Optimal threshold from PR curve (max F1): {optimal_threshold_pr_lgbm:.4f} "
                        f"(P: {precision_curve_lgbm[optimal_idx_pr_lgbm]:.4f}, R: {recall_curve_lgbm[optimal_idx_pr_lgbm]:.4f}, F1: {f1_scores_on_pr_curve_lgbm[optimal_idx_pr_lgbm]:.4f})")
        else:
            logger.warning("Optimal index from PR F1 out of bounds. Using last valid/default."); optimal_threshold_pr_lgbm = pr_thresholds_lgbm[-1] if len(pr_thresholds_lgbm) > 0 else 0.5
    else: logger.warning("Could not calculate F1 on PR curve.")
else: logger.warning("PR thresholds array empty.");

plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR Curve ({log_filename_base})'); plt.legend(loc="best"); plt.grid(True)
pr_curve_plot_path_lgbm = PLOT_DIR / f"{log_filename_base}_pr_curve_detailed.png"
try: plt.savefig(pr_curve_plot_path_lgbm); logger.info(f"Detailed PR curve plot saved to: {pr_curve_plot_path_lgbm}"); plt.close()
except Exception as e: logger.error(f"Error saving PR curve plot: {e}")

# ---------------------------------------------------------------------------
# 5. Evaluate Model with an Optimal Threshold
# ---------------------------------------------------------------------------
logger.info(f"=== 5. Evaluating LightGBM Model with Optimal Threshold (Maximizing F1-score) ===")
final_optimal_threshold_lgbm = optimal_threshold_pr_lgbm
if not (0 < final_optimal_threshold_lgbm < 1):
    logger.warning(f"Optimal threshold from PR curve ({final_optimal_threshold_lgbm}) invalid, trying iterative max F1 threshold ({best_f1_threshold_lgbm_iter}).")
    final_optimal_threshold_lgbm = best_f1_threshold_lgbm_iter
if not (0 < final_optimal_threshold_lgbm < 1):
    logger.error("Optimal threshold for LightGBM could not be determined reliably. Using default 0.5.")
    final_optimal_threshold_lgbm = 0.5
else:
    logger.info(f"Using optimal threshold for LightGBM: {final_optimal_threshold_lgbm:.4f}")

y_pred_optimal_lgbm = (y_pred_proba_tuned_lgbm >= final_optimal_threshold_lgbm).astype(int)
logger.info(f"\n--- Evaluation Metrics for LightGBM at Optimal Threshold ({final_optimal_threshold_lgbm:.4f}) ---")
precision_opt_lgbm, recall_opt_lgbm, f1_opt_lgbm, _ = precision_recall_fscore_support(y_test, y_pred_optimal_lgbm, pos_label=1, average='binary', zero_division=0)
logger.info(f"Precision (at optimal threshold): {precision_opt_lgbm:.4f}")
logger.info(f"Recall (at optimal threshold):    {recall_opt_lgbm:.4f}")
logger.info(f"F1-score (at optimal threshold):  {f1_opt_lgbm:.4f}")
report_optimal_lgbm = classification_report(y_test, y_pred_optimal_lgbm, target_names=['Not Canceled (0)', 'Canceled (1)'], zero_division=0)
logger.info(f"\nClassification Report (LightGBM at optimal threshold):\n{report_optimal_lgbm}")
cm_optimal_lgbm = confusion_matrix(y_test, y_pred_optimal_lgbm)
logger.info(f"Confusion Matrix (LightGBM at optimal threshold):\n{cm_optimal_lgbm}")

optimal_metrics_path_lgbm = RESULTS_DIR / f"{log_filename_base}_evaluation_metrics_optimal_thresh.txt"
try:
    with open(optimal_metrics_path_lgbm, 'w') as f:
        f.write(f"--- Tuned LightGBM with Optimal Threshold ({final_optimal_threshold_lgbm:.4f}) ---\n")
        f.write(f"Precision: {precision_opt_lgbm:.4f}\nRecall:    {recall_opt_lgbm:.4f}\nF1-score:  {f1_opt_lgbm:.4f}\n\n")
        f.write(f"Underlying PR AUC of the model: {avg_precision_lgbm:.4f}\n")
        f.write(f"Classification Report:\n{report_optimal_lgbm}\n\nCM:\n{np.array2string(cm_optimal_lgbm, separator=', ')}\n")
    logger.info(f"Optimal threshold evaluation metrics (LightGBM) saved to: {optimal_metrics_path_lgbm}")
except Exception as e: logger.error(f"Error saving optimal threshold metrics for LightGBM: {e}")

# ---------------------------------------------------------------------------
# 6. Comparison Summary
# ---------------------------------------------------------------------------
logger.info("\n=== 6. Final Model Comparison Summary ===")
logger.info("Metrics for Tuned LightGBM with Optimal Threshold:")
logger.info(f"  LGBM OptThresh - P: {precision_opt_lgbm:.4f}, R: {recall_opt_lgbm:.4f}, F1: {f1_opt_lgbm:.4f} (Base PR AUC: {avg_precision_lgbm:.4f})")
logger.info("Compare with Tuned XGBoost with Optimal Threshold (from 007_Model_XGBoost_Threshold_Tuning):")
logger.info("  (Check file '007_Model_XGBoost_Threshold_Tuning_evaluation_metrics_optimal_thresh.txt' for XGBoost optimal threshold results)")

logger.info(f"\n--- Threshold Tuning for Tuned LightGBM ({log_filename_base}) complete. Full output saved to: {log_filepath.resolve()} ---")

if __name__ == "__main__":
    pass