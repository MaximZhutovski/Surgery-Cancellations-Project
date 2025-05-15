# Appendix_A_Alternative_Classifiers.py
"""
Trains and evaluates Random Forest and Logistic Regression models.
Compares their performance to the tuned XGBoost model.

Run from the project root:
    python -m scripts.008_Alternative_Models
or from *scripts/* directly:
    python 008_Alternative_Models.py
"""
import sys
import datetime
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
                        PROCESSED_FEATURE_NAMES_PATH)
except ImportError as e:
    print(f"Error importing from config: {e}")
    print("Using default relative paths assuming script is in 'scripts' folder.")
    # (Fallback paths - similar to previous scripts)
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
logger.info(f"--- Alternative Models: Random Forest & Logistic Regression (Script 008) ---")

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
    logger.info(f"Loaded y_train: {y_train.shape}")

except FileNotFoundError as e:
    logger.error(f"Error: A required data file was not found: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"An error occurred while loading data: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Helper function for model evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, model_name, X_test, y_test, feature_names):
    logger.info(f"\n--- Evaluating: {model_name} ---")
    start_time = time.time()
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        logger.error(f"Error during prediction with {model_name}: {e}")
        return None
    eval_time = time.time() - start_time
    logger.info(f"Prediction time: {eval_time:.2f} seconds")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-score:  {f1:.4f}")
    logger.info(f"ROC AUC:   {roc_auc:.4f}")
    logger.info(f"PR AUC:    {pr_auc:.4f}")

    report = classification_report(y_test, y_pred, target_names=['Not Canceled (0)', 'Canceled (1)'])
    logger.info(f"\nClassification Report for {model_name}:\n{report}")

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix for {model_name}:\n{cm}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Predicted Not Canceled', 'Predicted Canceled'],
                yticklabels=['Actual Not Canceled', 'Actual Canceled'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    cm_plot_path = PLOT_DIR / f"008_confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    try:
        plt.savefig(cm_plot_path)
        logger.info(f"Confusion matrix plot for {model_name} saved to: {cm_plot_path}")
        plt.close()
    except Exception as e:
        logger.error(f"Error saving confusion matrix plot for {model_name}: {e}")

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        fi_df = fi_df.sort_values(by='importance', ascending=False).head(20)
        logger.info(f"\nTop 20 Feature Importances for {model_name}:")
        logger.info(fi_df.to_string(index=False))
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=fi_df, palette='coolwarm')
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.tight_layout()
        fi_plot_path = PLOT_DIR / f"008_feature_importance_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(fi_plot_path)
        logger.info(f"Feature importance plot for {model_name} saved to: {fi_plot_path}")
        plt.close()

    elif hasattr(model, 'coef_'): # For Logistic Regression
        # For Logistic Regression, coefficients can be interpreted as importance
        # (absolute value for magnitude, sign for direction)
        # Since we have many OHE features, this might be very long.
        # We'll take absolute values for magnitude ranking.
        if X_test.shape[1] == len(feature_names): # Ensure feature names match columns
            coefs = model.coef_[0] # coef_ is 2D for binary, take the first row
            fi_df = pd.DataFrame({'feature': feature_names, 'coefficient_abs': np.abs(coefs), 'coefficient': coefs})
            fi_df = fi_df.sort_values(by='coefficient_abs', ascending=False).head(20)
            logger.info(f"\nTop 20 Features by Absolute Coefficient for {model_name}:")
            logger.info(fi_df[['feature', 'coefficient']].to_string(index=False))

            plt.figure(figsize=(10, 8))
            # Sort by actual coefficient value for plotting to see positive/negative influences
            fi_df_plot = fi_df.copy()
            fi_df_plot['positive'] = fi_df_plot['coefficient'] > 0
            fi_df_plot = fi_df_plot.sort_values(by='coefficient_abs', ascending=True) # For horizontal bar plot
            
            sns.barplot(x='coefficient', y='feature', data=fi_df_plot, hue='positive', dodge=False, palette={True: "g", False: "r"})
            plt.title(f'Top 20 Feature Coefficients - {model_name}')
            plt.tight_layout()
            fi_plot_path = PLOT_DIR / f"008_coefficients_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(fi_plot_path)
            logger.info(f"Coefficient plot for {model_name} saved to: {fi_plot_path}")
            plt.close()
        else:
            logger.warning(f"Mismatch in feature names length and X_test columns for {model_name}. Skipping coefficient plot.")


    # Save evaluation metrics to a text file
    metrics_path = RESULTS_DIR / f"008_evaluation_metrics_{model_name.lower().replace(' ', '_')}.txt"
    try:
        with open(metrics_path, 'w') as f:
            f.write(f"--- {model_name} Evaluation Metrics ---\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1-score:  {f1:.4f}\n")
            f.write(f"ROC AUC:   {roc_auc:.4f}\n")
            f.write(f"PR AUC:    {pr_auc:.4f}\n\n")
            f.write(f"Classification Report:\n{report}\n\n")
            f.write(f"Confusion Matrix:\n{np.array2string(cm, separator=', ')}\n")
        logger.info(f"Evaluation metrics for {model_name} saved to: {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving metrics for {model_name}: {e}")
        
    return model # Return the trained model

# ---------------------------------------------------------------------------
# 2. Train and Evaluate Random Forest
# ---------------------------------------------------------------------------
logger.info("\n=== 2. Training and Evaluating Random Forest ===")
# Using class_weight='balanced_subsample' is often good for imbalanced RF
# n_estimators can be increased, max_depth can be tuned.
rf_model = RandomForestClassifier(
    n_estimators=200,       # Number of trees
    max_depth=10,           # Limit depth to prevent overfitting (can be tuned)
    random_state=42,
    class_weight='balanced_subsample', # Handles imbalance
    n_jobs=-1,
    min_samples_split=10,    # Min samples to split a node
    min_samples_leaf=5       # Min samples in a leaf node
)

logger.info("Training Random Forest model...")
start_time = time.time()
try:
    rf_model.fit(X_train_processed, y_train)
except Exception as e:
    logger.error(f"Error during Random Forest training: {e}")
    sys.exit(1)
train_time_rf = time.time() - start_time
logger.info(f"Random Forest training completed in {train_time_rf:.2f} seconds.")

trained_rf_model = evaluate_model(rf_model, "RandomForest", X_test_processed, y_test, processed_feature_names)

if trained_rf_model:
    RF_MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
    try:
        joblib.dump(trained_rf_model, RF_MODEL_PATH)
        logger.info(f"Trained Random Forest model saved to: {RF_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error saving Random Forest model: {e}")

# ---------------------------------------------------------------------------
# 3. Train and Evaluate Logistic Regression
# ---------------------------------------------------------------------------
logger.info("\n=== 3. Training and Evaluating Logistic Regression ===")
# Data is already scaled from script 004
# Using class_weight='balanced'
# Solver 'liblinear' is good for smaller datasets and supports L1/L2.
# 'saga' is good for larger datasets and supports L1/L2/ElasticNet.
# max_iter might need adjustment for convergence.
lr_model = LogisticRegression(
    random_state=42,
    class_weight='balanced', # Handles imbalance
    solver='liblinear',      # Good for L1/L2, smaller datasets
    penalty='l1',            # L1 regularization for feature selection/sparsity
    C=0.1,                   # Inverse of regularization strength; smaller C = stronger regularization
    max_iter=1000            # Increased max_iter
)

logger.info("Training Logistic Regression model...")
start_time = time.time()
try:
    lr_model.fit(X_train_processed, y_train)
except Exception as e:
    logger.error(f"Error during Logistic Regression training: {e}")
    sys.exit(1)
train_time_lr = time.time() - start_time
logger.info(f"Logistic Regression training completed in {train_time_lr:.2f} seconds.")

trained_lr_model = evaluate_model(lr_model, "LogisticRegression", X_test_processed, y_test, processed_feature_names)

if trained_lr_model:
    LR_MODEL_PATH = MODEL_DIR / "logistic_regression_model.joblib"
    try:
        joblib.dump(trained_lr_model, LR_MODEL_PATH)
        logger.info(f"Trained Logistic Regression model saved to: {LR_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error saving Logistic Regression model: {e}")

# ---------------------------------------------------------------------------
# 4. Summary and Next Steps
# ---------------------------------------------------------------------------
logger.info("\n=== 4. Summary and Next Steps ===")
logger.info("Random Forest and Logistic Regression models have been trained and evaluated.")
logger.info("Compare their performance (Precision, Recall, F1, PR AUC for 'Canceled' class) with the tuned XGBoost model from script 006.")
logger.info("Considerations for choosing a model:")
logger.info("  - Performance on key metrics (PR AUC, F1-score for the minority class).")
logger.info("  - Training and prediction time.")
logger.info("  - Interpretability (Logistic Regression coefficients are more directly interpretable).")
logger.info("  - Complexity of tuning and maintenance.")
logger.info("Next steps could involve further tuning of these models or exploring more advanced techniques.")

# --- End of Script ---
logger.info(f"\n--- Alternative Models training and evaluation complete. Full output saved to: {log_filepath.resolve()} ---")

if __name__ == "__main__":
    pass