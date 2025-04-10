# =========================================
# Feature Engineering for Surgery Cancellation
# =========================================

# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
from geopy.distance import geodesic

# --- Step 2: Load Data ---
file_path = r"C:\Users\coolm\Desktop\לימודים\שנה ג\Surgery_Cancellation\data\surgery_data.xlsx"
xls = pd.ExcelFile(file_path)
df_all = pd.read_excel(xls, sheet_name=0)
df_canceled = pd.read_excel(xls, sheet_name=1)

# Mark canceled surgeries in the main dataframe
df_all['was_canceled'] = df_all['מספר ניתוח תכנון'].isin(df_canceled['מספר ניתוח תכנון'])

# --- Step 3: Basic Features ---
# Weekday of surgery
df_all['surgery_weekday'] = pd.to_datetime(df_all['תאריך ביצוע ניתוח']).dt.day_name()

# Is surgery on weekend? (Friday/Saturday)
df_all['is_weekend'] = df_all['surgery_weekday'].isin(['Friday', 'Saturday'])

# Season of surgery
df_all['surgery_month'] = pd.to_datetime(df_all['תאריך ביצוע ניתוח']).dt.month
df_all['season'] = df_all['surgery_month'].map({
    12:'Winter', 1:'Winter', 2:'Winter',
    3:'Spring', 4:'Spring', 5:'Spring',
    6:'Summer', 7:'Summer', 8:'Summer',
    9:'Fall', 10:'Fall', 11:'Fall'
})

# --- Step 4: Age groups (by decade) ---
df_all['age_decade'] = (df_all['גיל המטופל בזמן הניתוח'] // 10) * 10
df_all['age_decade'] = df_all['age_decade'].astype('Int64')

# --- Step 5: BMI missing indicator ---
df_all['bmi_missing'] = df_all['BMI'].isnull()

# --- Step 6: Number of missing Lab tests ---
lab_cols = ['ALBUMIN', 'PT-INR', 'POTASSIUM', 'SODIUM', 'HB', 'WBC', 'PLT']
df_all['num_missing_labs'] = df_all[lab_cols].isnull().sum(axis=1)
df_all['has_missing_labs'] = df_all['num_missing_labs'] > 0

# --- Step 7: Number of Medications and Diagnoses ---
df_all['num_medications'] = df_all['תרופות קבועות'].apply(
    lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

df_all['num_diagnoses'] = df_all['אבחנות רקע'].apply(
    lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

# --- Step 8: Surgery Cancellation rates by Operating Room ---
room_cancel_rates = df_all.groupby('חדר')['was_canceled'].mean()
df_all['room_cancel_rate'] = df_all['חדר'].map(room_cancel_rates)

# --- Step 9: Surgery Cancellation rates by Anesthesia type ---
anesthesia_cancel_rates = df_all.groupby('הרדמה')['was_canceled'].mean()
df_all['anesthesia_cancel_rate'] = df_all['הרדמה'].map(anesthesia_cancel_rates)

# --- Step 10: Surgery Cancellation rates by Procedure ---
procedure_cancel_rates = df_all.groupby('קוד ניתוח')['was_canceled'].mean()
df_all['procedure_cancel_rate'] = df_all['קוד ניתוח'].map(procedure_cancel_rates)

# --- Step 11: Holidays indicator (Israel major holidays as example) ---
holidays_dates = pd.to_datetime([
    '2024-04-22', '2024-04-29',  # Passover
    '2024-05-14',                # Independence Day
    '2024-09-25', '2024-09-26',  # Rosh Hashanah
    '2024-10-04',                # Yom Kippur
    '2024-10-09',                # Sukkot
])

df_all['surgery_date'] = pd.to_datetime(df_all['תאריך ביצוע ניתוח'])
df_all['near_holiday'] = df_all['surgery_date'].apply(
    lambda x: any(abs((x - holiday).days) <= 3 for holiday in holidays_dates))

# --- Step 12: Distance from hospital (requires external data) ---
df_coords = pd.read_excel(r"C:\Users\coolm\Desktop\לימודים\שנה ג\Surgery_Cancellation\data\israel_city_coordinate.xlsx")

# Coordinates of the hospital (example: Tel Aviv Sourasky Medical Center
hospital_coords = (32.181928, 34.89585)

# Extracting city coordinates from the external data

city_coords = dict(zip(df_coords['city'], zip(df_coords['latitude'], df_coords['longitude'])))

def calculate_distance(city_name):
    try:
        city_coordinate = city_coords[city_name]
        distance = geodesic(hospital_coords, city_coordinate).km
        return distance
    except KeyError:
        return None  # במידה ואין התאמה לעיר, נחזיר None
    
df_all['distance_from_hospital_km'] = df_all['עיר'].apply(calculate_distance)

print(df_all[['עיר', 'distance_from_hospital_km']].head(20))

df_all.to_excel("Surgery_Data_with_Distance.xlsx", index=False)

# --- Final Check: Display engineered features ---
features_to_display = [
    'was_canceled', 'surgery_weekday', 'is_weekend', 'season',
    'age_decade', 'bmi_missing', 'num_missing_labs', 'has_missing_labs',
    'num_medications', 'num_diagnoses', 'room_cancel_rate',
    'anesthesia_cancel_rate', 'procedure_cancel_rate',
    'near_holiday', 'distance_from_hospital_km'
]

print(df_all[features_to_display].head(500))


# ==========================================
# Explanation of Engineered Features
# ==========================================

# 'was_canceled': Indicates whether the scheduled surgery was canceled (True) or not (False).

# 'surgery_weekday': The weekday (e.g., Sunday, Monday) on which the surgery was scheduled.

# 'is_weekend': Boolean indicator, True if surgery was scheduled on weekend (Friday/Saturday), otherwise False.

# 'season': Season of the year when the surgery was scheduled (Winter, Spring, Summer, Fall).

# 'age_decade': Age group of the patient categorized by decades (e.g., 40 represents ages 40-49).

# 'bmi_missing': Indicates if BMI data is missing for the patient (True if missing, False otherwise).

# 'num_missing_labs': Count of missing laboratory tests (from ALBUMIN, PT-INR, POTASSIUM, SODIUM, HB, WBC, PLT).

# 'has_missing_labs': Boolean indicator, True if at least one lab test is missing, False otherwise.

# 'num_medications': Number of regular medications that the patient is taking.

# 'num_diagnoses': Number of medical diagnoses recorded for the patient.

# 'room_cancel_rate': Historical cancellation rate of the specific operating room assigned.

# 'anesthesia_cancel_rate': Historical cancellation rate based on the type of anesthesia.

# 'procedure_cancel_rate': Historical cancellation rate based on the specific surgical procedure code.

# 'near_holiday': Boolean indicator, True if surgery was scheduled near major holidays (within ±3 days).

# 'distance_from_hospital_km': Distance (in kilometers) from patient's residential city to hospital. 
# Currently empty; requires external city coordinates data to populate.
