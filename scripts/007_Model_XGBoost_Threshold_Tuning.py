# 007_Model_XGBoost_Threshold_Tuning.py
"""
Performs threshold tuning for the tuned XGBoost model
(from script 006_Model_XGBoost_Tuning.py)
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
matplotlib.use('Agg')
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
                        X_TEST_PROCESSED_PATH, Y_TEST_PATH,   # Input data from 004
                        XGB_TUNED_MODEL_PATH)                 # Input model from 006
    print("Successfully imported paths from config.py for 007")
except ImportError as e:
    print(f"CRITICAL (007): Error importing from config.py: {e}")
    # Fallback paths
    scripts_dir = Path(__file__).resolve().parent; project_root_alt = scripts_dir.parent
    RESULTS_DIR = project_root_alt / "results"; DATA_DIR = project_root_alt / "data"
    PLOT_DIR = project_root_alt / "plots"; MODEL_DIR = DATA_DIR / "models"
    print(f"Warning (007): Using fallback paths.")
    X_TEST_PROCESSED_PATH = DATA_DIR / "004_X_test_processed.joblib"
    Y_TEST_PATH = DATA_DIR / "004_y_test.joblib"
    XGB_TUNED_MODEL_PATH = MODEL_DIR / "006_xgb_tuned_model.joblib"
    for d_path in (DATA_DIR, PLOT_DIR, RESULTS_DIR, MODEL_DIR): d_path.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"CRITICAL (007): An unexpected error occurred during config import or path init: {e}")
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
log_filename_base = Path(__file__).stem # Should be "007_Model_XGBoost_Threshold_Tuning"
log_filepath = get_next_filename(RESULTS_DIR, log_filename_base, suffix=".txt")
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(); logger.handlers.clear(); logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(log_filepath, encoding='utf-8'); file_handler.setFormatter(log_formatter); logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout); console_handler.setFormatter(log_formatter); logger.addHandler(console_handler)

# --- Start of Script ---
logger.info(f"--- XGBoost Threshold Tuning (Script {log_filename_base}) ---")
logger.info(f"Using tuned model from: {XGB_TUNED_MODEL_PATH}")
logger.info(f"Using test data from: {X_TEST_PROCESSED_PATH}")

# ---------------------------------------------------------------------------
# 1. Load Processed Test Data and Tuned XGBoost Model
# ---------------------------------------------------------------------------
logger.info("=== 1. Loading Processed Test Data and Tuned XGBoost Model ===")
try:
    X_test_processed = joblib.load(X_TEST_PROCESSED_PATH)
    y_test = joblib.load(Y_TEST_PATH)
    if not XGB_TUNED_MODEL_PATH.exists():
        logger.error(f"Tuned XGBoost model file not found at: {XGB_TUNED_MODEL_PATH}")
        logger.error(f"Please ensure script 006 (e.g., 006_Model_XGBoost_Tuning.py) ran successfully and saved the model.")
        sys.exit(1)
    tuned_xgb_model = joblib.load(XGB_TUNED_MODEL_PATH)
    logger.info(f"Loaded X_test_processed: {X_test_processed.shape}")
    logger.info(f"Tuned XGBoost model loaded successfully from: {XGB_TUNED_MODEL_PATH}")
except FileNotFoundError as e:
    logger.error(f"Error: A required data/model file was not found: {e}"); sys.exit(1)
except Exception as e:
    logger.error(f"An error occurred while loading data/model: {e}"); sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Predict Probabilities
# ---------------------------------------------------------------------------
logger.info("=== 2. Predicting Probabilities on Test Set (Tuned XGBoost) ===")
try:
    y_pred_proba_tuned_xgb = tuned_xgb_model.predict_proba(X_test_processed)[:, 1]
    logger.info("Probabilities predicted successfully.")
except Exception as e:
    logger.error(f"Error during probability prediction: {e}"); sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Iterate Through Thresholds and Calculate Metrics
# ---------------------------------------------------------------------------
logger.info("=== 3. Evaluating Metrics for Different Thresholds (Tuned XGBoost) ===")
thresholds = np.arange(0.05, 0.96, 0.01)
results_xgb = []
for thresh in thresholds:
    y_pred_thresh_xgb = (y_pred_proba_tuned_xgb >= thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_thresh_xgb, pos_label=1, average='binary', zero_division=0)
    results_xgb.append({'threshold': thresh, 'precision': precision, 'recall': recall, 'f1_score': f1})
results_df_xgb = pd.DataFrame(results_xgb)

logger.info("\nMetrics for various thresholds (sample - tuned XGBoost):")
logger.info(results_df_xgb.iloc[::10].to_string(index=False)) # Print every 10th row
threshold_results_csv_path_xgb = RESULTS_DIR / f"{log_filename_base}_threshold_tuning_results.csv"
results_df_xgb.to_csv(threshold_results_csv_path_xgb, index=False)
logger.info(f"\nFull threshold tuning results (XGBoost) saved to: {threshold_results_csv_path_xgb}")

# ---------------------------------------------------------------------------
# 4. Plot Precision, Recall, and F1-score vs. Threshold
# ---------------------------------------------------------------------------
logger.info("=== 4. Plotting Metrics vs. Threshold (Tuned XGBoost) ===")
plt.figure(figsize=(12, 7))
plt.plot(results_df_xgb['threshold'], results_df_xgb['precision'], label='Precision', color='blue', marker='.')
plt.plot(results_df_xgb['threshold'], results_df_xgb['recall'], label='Recall', color='green', marker='.')
plt.plot(results_df_xgb['threshold'], results_df_xgb['f1_score'], label='F1-score', color='red', marker='.')
plt.title(f'P, R, F1 vs. Threshold ({log_filename_base})')
plt.xlabel('Threshold'); plt.ylabel('Score'); plt.legend(); plt.grid(True); plt.xticks(np.arange(0.0, 1.01, 0.05))
plt.axvline(x=0.5, color='grey', linestyle='--', label='Default Thresh (0.5)')

best_f1_row_xgb = results_df_xgb.loc[results_df_xgb['f1_score'].idxmax()] if not results_df_xgb.empty else None
best_f1_threshold_xgb_iter = 0.5 # Default
if best_f1_row_xgb is not None:
    best_f1_threshold_xgb_iter = best_f1_row_xgb['threshold']
    max_f1_score_xgb_iter = best_f1_row_xgb['f1_score']
    plt.axvline(x=best_f1_threshold_xgb_iter, color='red', linestyle=':', label=f'Max F1 Iter ({best_f1_threshold_xgb_iter:.3f})')
    logger.info(f"\nThreshold for Max F1-score (from iteration): {best_f1_threshold_xgb_iter:.3f} (F1: {max_f1_score_xgb_iter:.4f})")
else:
    logger.warning("Could not determine best F1 threshold from iteration for XGBoost.")

plt.legend()
threshold_plot_path_xgb = PLOT_DIR / f"{log_filename_base}_metrics_vs_threshold_plot.png"
try: plt.savefig(threshold_plot_path_xgb); logger.info(f"Threshold metrics plot saved to: {threshold_plot_path_xgb}"); plt.close()
except Exception as e: logger.error(f"Error saving threshold metrics plot: {e}")

precision_curve_xgb, recall_curve_xgb, pr_thresholds_xgb = precision_recall_curve(y_test, y_pred_proba_tuned_xgb)
avg_precision_xgb = average_precision_score(y_test, y_pred_proba_tuned_xgb)
plt.figure(figsize=(10, 7)); plt.plot(recall_curve_xgb, precision_curve_xgb, lw=2, color='darkorange', label=f'PR curve (AUC = {avg_precision_xgb:.3f})')
optimal_threshold_pr_xgb = 0.5 # Default
if len(pr_thresholds_xgb) > 0 :
    f1_scores_on_pr_curve_xgb = (2 * precision_curve_xgb[:-1] * recall_curve_xgb[:-1]) / (precision_curve_xgb[:-1] + recall_curve_xgb[:-1] + 1e-9)
    if len(f1_scores_on_pr_curve_xgb) > 0:
        optimal_idx_pr_xgb = np.argmax(f1_scores_on_pr_curve_xgb)
        if optimal_idx_pr_xgb < len(pr_thresholds_xgb):
            optimal_threshold_pr_xgb = pr_thresholds_xgb[optimal_idx_pr_xgb]
            plt.scatter(recall_curve_xgb[optimal_idx_pr_xgb], precision_curve_xgb[optimal_idx_pr_xgb], marker='o', color='red', s=100,
                        label=f'Best F1 (Thresh≈{optimal_threshold_pr_xgb:.3f})')
            logger.info(f"Optimal threshold from PR curve (max F1): {optimal_threshold_pr_xgb:.4f} "
                        f"(P: {precision_curve_xgb[optimal_idx_pr_xgb]:.4f}, R: {recall_curve_xgb[optimal_idx_pr_xgb]:.4f}, F1: {f1_scores_on_pr_curve_xgb[optimal_idx_pr_xgb]:.4f})")
        else:
            logger.warning("Optimal index from PR F1 out of bounds. Using last valid/default."); optimal_threshold_pr_xgb = pr_thresholds_xgb[-1] if len(pr_thresholds_xgb) > 0 else 0.5
    else: logger.warning("Could not calculate F1 on PR curve.");
else: logger.warning("PR thresholds array empty.");

plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR Curve ({log_filename_base})'); plt.legend(loc="best"); plt.grid(True)
pr_curve_plot_path_xgb = PLOT_DIR / f"{log_filename_base}_pr_curve_detailed.png"
try: plt.savefig(pr_curve_plot_path_xgb); logger.info(f"Detailed PR curve plot saved to: {pr_curve_plot_path_xgb}"); plt.close()
except Exception as e: logger.error(f"Error saving PR curve plot: {e}")

# ---------------------------------------------------------------------------
# 5. Evaluate Model with an Optimal Threshold
# ---------------------------------------------------------------------------
logger.info(f"=== 5. Evaluating XGBoost Model with Optimal Threshold (Maximizing F1-score) ===")
final_optimal_threshold_xgb = optimal_threshold_pr_xgb
if not (0 < final_optimal_threshold_xgb < 1): # Check if it's a valid probability
    logger.warning(f"Optimal threshold from PR curve ({final_optimal_threshold_xgb}) invalid, trying iterative max F1 threshold ({best_f1_threshold_xgb_iter}).")
    final_optimal_threshold_xgb = best_f1_threshold_xgb_iter
if not (0 < final_optimal_threshold_xgb < 1):
    logger.error("Optimal threshold for XGBoost could not be determined reliably. Using default 0.5.")
    final_optimal_threshold_xgb = 0.5
else:
    logger.info(f"Using optimal threshold for XGBoost: {final_optimal_threshold_xgb:.4f}")

y_pred_optimal_xgb = (y_pred_proba_tuned_xgb >= final_optimal_threshold_xgb).astype(int)
logger.info(f"\n--- Evaluation Metrics for XGBoost at Optimal Threshold ({final_optimal_threshold_xgb:.4f}) ---")
precision_opt_xgb, recall_opt_xgb, f1_opt_xgb, _ = precision_recall_fscore_support(y_test, y_pred_optimal_xgb, pos_label=1, average='binary', zero_division=0)

logger.info(f"Precision (at optimal threshold): {precision_opt_xgb:.4f}")
logger.info(f"Recall (at optimal threshold):    {recall_opt_xgb:.4f}")
logger.info(f"F1-score (at optimal threshold):  {f1_opt_xgb:.4f}")

report_optimal_xgb = classification_report(y_test, y_pred_optimal_xgb, target_names=['Not Canceled (0)', 'Canceled (1)'], zero_division=0)
logger.info(f"\nClassification Report (XGBoost at optimal threshold):\n{report_optimal_xgb}")
cm_optimal_xgb = confusion_matrix(y_test, y_pred_optimal_xgb)
logger.info(f"Confusion Matrix (XGBoost at optimal threshold):\n{cm_optimal_xgb}")

optimal_metrics_path_xgb = RESULTS_DIR / f"{log_filename_base}_evaluation_metrics_optimal_thresh.txt"
try:
    with open(optimal_metrics_path_xgb, 'w') as f:
        f.write(f"--- Tuned XGBoost with Optimal Threshold ({final_optimal_threshold_xgb:.4f}) ---\n")
        f.write(f"Precision: {precision_opt_xgb:.4f}\nRecall:    {recall_opt_xgb:.4f}\nF1-score:  {f1_opt_xgb:.4f}\n\n")
        f.write(f"Underlying PR AUC of the model: {avg_precision_xgb:.4f}\n")
        f.write(f"Classification Report:\n{report_optimal_xgb}\n\nCM:\n{np.array2string(cm_optimal_xgb, separator=', ')}\n")
    logger.info(f"Optimal threshold evaluation metrics (XGBoost) saved to: {optimal_metrics_path_xgb}")
except Exception as e: logger.error(f"Error saving optimal threshold metrics for XGBoost: {e}")

logger.info(f"\n--- Threshold Tuning for Tuned XGBoost ({log_filename_base}) complete. Full output saved to: {log_filepath.resolve()} ---")

if __name__ == "__main__":
    pass