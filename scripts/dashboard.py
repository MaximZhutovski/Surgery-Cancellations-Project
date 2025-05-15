import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re # For parsing text files

# --- Configuration (Import paths from config.py) ---
# This assumes dashboard.py is in the project root, and 'scripts' is a subdirectory.
# If dashboard.py is inside 'scripts', adjust project_root.
try:
    # Try to make it runnable from project root or scripts directory
    current_dir = Path(__file__).resolve().parent
    if (current_dir / "config.py").exists(): # If config is in the same dir as dashboard
         from config import (RESULTS_DIR, PLOT_DIR, MODEL_DIR) # Add more as needed
    elif (current_dir / "scripts" / "config.py").exists(): # If dashboard is in project root
        import sys
        sys.path.insert(0, str(current_dir / "scripts"))
        from config import (RESULTS_DIR, PLOT_DIR, MODEL_DIR)
    else: # Fallback if structure is different
        RESULTS_DIR = Path("results")
        PLOT_DIR = Path("plots")
        MODEL_DIR = Path("models")
        st.warning("Could not automatically determine config.py location. Using default relative paths for results, plots, models.")
    
    # Define paths to specific result files based on the new script numbering
    # XGBoost Tuned with Optimal Threshold (from 007)
    xgb_opt_metrics_file = RESULTS_DIR / "007_Model_XGBoost_Threshold_Tuning_evaluation_metrics_optimal_thresh.txt"
    xgb_threshold_csv_file = RESULTS_DIR / "007_Model_XGBoost_Threshold_Tuning_threshold_tuning_results.csv"
    xgb_cm_opt_plot = PLOT_DIR / "007_Model_XGBoost_Threshold_Tuning_CM.png" # Assuming CM for optimal is saved like this or named specifically
    xgb_fi_tuned_plot = PLOT_DIR / "006_Model_XGBoost_Tuning_FI.png" # FI from the tuned model before thresholding
    xgb_pr_thresh_plot = PLOT_DIR / "007_Model_XGBoost_Threshold_Tuning_metrics_vs_threshold_plot.png"

    # LightGBM Tuned with Optimal Threshold (from 009)
    lgbm_opt_metrics_file = RESULTS_DIR / "009_Model_LightGBM_Threshold_Tuning_evaluation_metrics_optimal_thresh.txt"
    lgbm_threshold_csv_file = RESULTS_DIR / "009_Model_LightGBM_Threshold_Tuning_threshold_tuning_results.csv"
    lgbm_cm_opt_plot = PLOT_DIR / "009_Model_LightGBM_Threshold_Tuning_CM.png" # Assuming CM for optimal
    lgbm_fi_tuned_plot = PLOT_DIR / "008_Model_LightGBM_Optuna_Tuning_FI.png" # FI from the tuned model before thresholding
    lgbm_pr_thresh_plot = PLOT_DIR / "009_Model_LightGBM_Threshold_Tuning_metrics_vs_threshold_plot.png"

    # Baseline XGB (from 005)
    xgb_baseline_metrics_file = RESULTS_DIR / "005_Model_XGBoost_Baseline_evaluation_metrics.txt"
    
    # (Optional) RF and LR from Appendix/008
    rf_metrics_file = RESULTS_DIR / "Appendix_A_Alternative_Classifiers_evaluation_metrics_randomforest.txt" # Adjust name if different
    lr_metrics_file = RESULTS_DIR / "Appendix_A_Alternative_Classifiers_evaluation_metrics_logisticregression.txt" # Adjust name if different

except ImportError:
    st.error("CRITICAL: config.py not found. Please ensure it's in the 'scripts' directory or project root and paths are correct.")
    # Define fallback paths if config import fails, so the app can still try to run
    RESULTS_DIR = Path("results")
    PLOT_DIR = Path("plots")
    MODEL_DIR = Path("models")
    # Define fallback for specific files (these will likely fail if config.py is missing)
    xgb_opt_metrics_file = RESULTS_DIR / "007_Model_XGBoost_Threshold_Tuning_evaluation_metrics_optimal_thresh.txt"
    lgbm_opt_metrics_file = RESULTS_DIR / "009_Model_LightGBM_Threshold_Tuning_evaluation_metrics_optimal_thresh.txt"
    # ... and so on for other files
    st.stop() # Stop execution if config cannot be loaded properly


st.set_page_config(layout="wide")
st.title("ניתוח וחיזוי ביטולי ניתוחים - דאשבורד תוצאות")

# --- Helper function to parse metrics from text files ---
def parse_metrics_from_file(file_path):
    metrics = {"Precision": np.nan, "Recall": np.nan, "F1-score": np.nan, "PR AUC": np.nan, "ROC AUC": np.nan}
    classification_report_str = "Classification Report not found."
    confusion_matrix_str = "Confusion Matrix not found."
    
    if not file_path.exists():
        st.warning(f"Metrics file not found: {file_path}")
        return metrics, classification_report_str, confusion_matrix_str

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

            precision_match = re.search(r"Precision: (\d\.\d+)", content)
            if precision_match: metrics["Precision"] = float(precision_match.group(1))
            
            # For "Canceled (1)" class from classification report if available
            # This is more robust if the simple "Precision: X.XXXX" line changes
            report_match = re.search(r"Canceled \(1\)\s+(\d\.\d+)\s+(\d\.\d+)\s+(\d\.\d+)", content)
            if report_match:
                metrics["Precision"] = float(report_match.group(1))
                metrics["Recall"] = float(report_match.group(2))
                metrics["F1-score"] = float(report_match.group(3))

            pr_auc_match = re.search(r"PR AUC: (\d\.\d+)", content)
            if not pr_auc_match: # Try "Underlying PR AUC" for optimal threshold files
                 pr_auc_match = re.search(r"Underlying PR AUC of the model: (\d\.\d+)", content)
            if pr_auc_match: metrics["PR AUC"] = float(pr_auc_match.group(1))

            roc_auc_match = re.search(r"ROC AUC: (\d\.\d+)", content)
            if roc_auc_match: metrics["ROC AUC"] = float(roc_auc_match.group(1))

            report_start = content.find("Classification Report:")
            report_end = content.find("CM:") if content.find("CM:") != -1 else content.find("Confusion Matrix:")
            if report_start != -1 and report_end != -1:
                classification_report_str = content[report_start:report_end].strip()
            
            cm_start = content.find("CM:") if content.find("CM:") != -1 else content.find("Confusion Matrix:")
            if cm_start != -1:
                # Extract lines after CM: until next blank line or end of section
                cm_lines = content[cm_start:].split('\n')
                cm_data_str = ""
                for line in cm_lines[1:]: # Skip "CM:" line
                    if not line.strip(): break # Stop at first blank line
                    cm_data_str += line + "\n"
                confusion_matrix_str = cm_data_str.strip()

    except Exception as e:
        st.error(f"Error parsing metrics from {file_path}: {e}")
    return metrics, classification_report_str, confusion_matrix_str

# --- Load data for summary table ---
@st.cache_data # Cache the data loading
def load_all_metrics():
    models_data = []
    
    # XGBoost Tuned Optimal Threshold
    metrics_xgb_opt, _, _ = parse_metrics_from_file(xgb_opt_metrics_file)
    models_data.append({"Model": "XGBoost (Tuned, Opt Thresh)", **metrics_xgb_opt})

    # LightGBM Tuned Optimal Threshold
    metrics_lgbm_opt, _, _ = parse_metrics_from_file(lgbm_opt_metrics_file)
    models_data.append({"Model": "LightGBM (Tuned, Opt Thresh)", **metrics_lgbm_opt})

    # XGBoost Baseline (Untuned, New Feats)
    metrics_xgb_base, _, _ = parse_metrics_from_file(xgb_baseline_metrics_file)
    models_data.append({"Model": "XGBoost (Baseline, New Feats)", **metrics_xgb_base})
    
    # (Optional) Add RF and LR if files exist and are parsed
    if rf_metrics_file.exists():
        metrics_rf, _, _ = parse_metrics_from_file(rf_metrics_file)
        models_data.append({"Model": "Random Forest (Appendix A)", **metrics_rf})
    if lr_metrics_file.exists():
        metrics_lr, _, _ = parse_metrics_from_file(lr_metrics_file)
        models_data.append({"Model": "Logistic Regression (Appendix A)", **metrics_lr})

    return pd.DataFrame(models_data)

all_metrics_df = load_all_metrics()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["סקירה כללית", "השוואת מודלים", "פירוט XGBoost (אופטימלי)", "פירוט LightGBM (אופטימלי)"])

with tab1:
    st.header("סקירה כללית של הפרויקט")
    st.markdown("""
    **מטרת הפרויקט:** פיתוח מודל לחיזוי ביטולי תורים לניתוחים בטווח זמן קצר לפני מועד הניתוח.
    
    **הנתונים:** התבססו על נתונים היסטוריים של קביעת וביצוע ניתוחים.
    
    **התהליך כלל:**
    1.  ניתוח נתונים ראשוני (EDA).
    2.  הנדסת תכונות מקיפה ליצירת משתנים חדשים בעלי פוטנציאל חיזוי.
    3.  עיבוד מקדים של הנתונים (ניקוי, השלמת חסרים, קידוד, סקלור).
    4.  אימון והערכה של מגוון מודלים, כולל:
        *   XGBoost (עם כוונון היפרפרמטרים וכוונון סף החלטה).
        *   LightGBM (עם כוונון היפרפרמטרים באמצעות Optuna וכוונון סף החלטה).
        *   מודלים נוספים להשוואה (Random Forest, Logistic Regression).
    5.  בחינת טכניקות לטיפול בחוסר איזון במחלקות (למשל, `scale_pos_weight`, ניסויים עם SMOTE ו-RandomOverSampler).
        
    **הדאשבורד מציג את תוצאות המודלים העיקריים והשוואה ביניהם.**
    """)

with tab2:
    st.header("השוואת ביצועי מודלים (בדגש על חיזוי ביטולים - Class 1)")
    if not all_metrics_df.empty:
        st.dataframe(all_metrics_df.set_index("Model").style.format("{:.4f}", na_rep="-").highlight_max(axis=0, subset=['Precision', 'Recall', 'F1-score', 'PR AUC', 'ROC AUC'], color='lightgreen'))
        st.markdown("""
        *   **Precision (Canceled):** מתוך כל הניתוחים שהמודל חזה שיבוטלו, איזה אחוז מהם אכן בוטלו? (TP / (TP + FP))
        *   **Recall (Canceled):** מתוך כל הניתוחים שאכן בוטלו, איזה אחוז מהם המודל הצליח לזהות? (TP / (TP + FN))
        *   **F1-score (Canceled):** ממוצע הרמוני של Precision ו-Recall. מדד מאוזן.
        *   **PR AUC (Base):** שטח מתחת לעקומת Precision-Recall של המודל לפני התאמת סף. מדד טוב לבעיות לא מאוזנות.
        *   **ROC AUC:** שטח מתחת לעקומת ROC.
        """)
    else:
        st.warning("לא נטענו נתוני מדדים להשוואה.")

def display_model_details(model_name, metrics_file, cm_plot_file, fi_plot_file, threshold_csv_file, threshold_plot_file):
    st.header(f"פירוט עבור: {model_name}")
    
    metrics, report_str, cm_str = parse_metrics_from_file(metrics_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("מדדי ביצוע עיקריים (על סט המבחן):")
        if metrics:
            st.write(f"- Precision (Canceled): {metrics.get('Precision', 'N/A'):.4f}")
            st.write(f"- Recall (Canceled):    {metrics.get('Recall', 'N/A'):.4f}")
            st.write(f"- F1-score (Canceled):  {metrics.get('F1-score', 'N/A'):.4f}")
            st.write(f"- PR AUC (Base Model):  {metrics.get('PR AUC', 'N/A'):.4f}") # This is PR AUC of the base model
            st.write(f"- ROC AUC (Base Model): {metrics.get('ROC AUC', 'N/A'):.4f}")
    
    with col2:
        st.subheader("Confusion Matrix (סף אופטימלי):")
        if cm_plot_file.exists():
            st.image(str(cm_plot_file))
        else:
            st.text("Confusion Matrix:")
            st.text(cm_str if cm_str else "לא זמין")

    st.subheader("Classification Report המלא (סף אופטימלי):")
    st.text(report_str if report_str else "לא זמין")

    st.subheader("חשיבות תכונות (Top 20 מהמודל המכוונן):")
    if fi_plot_file.exists():
        st.image(str(fi_plot_file))
    else:
        st.warning(f"גרף חשיבות תכונות לא נמצא: {fi_plot_file}")
        
    st.subheader("גרף מדדים מול סף החלטה:")
    if threshold_plot_file.exists():
        st.image(str(threshold_plot_file))
    elif threshold_csv_file.exists():
        st.write(f"קובץ CSV עם נתוני ספים נמצא ({threshold_csv_file}), אך הגרף לא. ניתן לטעון ולהציג את ה-CSV.")
        # df_thresh = pd.read_csv(threshold_csv_file)
        # st.dataframe(df_thresh)
    else:
        st.warning(f"גרף או נתוני כוונון סף לא נמצאו: {threshold_plot_file} / {threshold_csv_file}")


with tab3:
    display_model_details("XGBoost (מכוונן, סף אופטימלי)", 
                          xgb_opt_metrics_file, 
                          xgb_cm_opt_plot, 
                          xgb_fi_tuned_plot, 
                          xgb_threshold_csv_file,
                          xgb_pr_thresh_plot)

with tab4:
    display_model_details("LightGBM (מכוונן Optuna, סף אופטימלי)", 
                          lgbm_opt_metrics_file, 
                          lgbm_cm_opt_plot, 
                          lgbm_fi_tuned_plot,
                          lgbm_threshold_csv_file,
                          lgbm_pr_thresh_plot)

st.sidebar.info("דאשבורד זה מסכם את תוצאות פרויקט חיזוי ביטולי ניתוחים.")