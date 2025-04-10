import pandas as pd

# === Step 1: Load Excel data ===
file_path = r"C:\Users\coolm\Desktop\לימודים\שנה ג\Surgery_Cancellation\data\surgery_data.xlsx"
xls = pd.ExcelFile(file_path)
df_all = pd.read_excel(xls, sheet_name=0)
df_canceled = pd.read_excel(xls, sheet_name=1)

# === Step 2: Basic Overview ===
print("\n=== Data Overview ===")
print(f"Total records (All Surgeries): {len(df_all)}")
print(f"Total records (Canceled Surgeries): {len(df_canceled)}")
print(f"Unique patients (All): {df_all['ת\"ז מותממת'].nunique()}")
print(f"Unique patients who canceled: {df_canceled['ת\"ז מותממת'].nunique()}")

# === Step 3: Missing Values Analysis for All Surgeries ===
missing_all = df_all.isnull().mean() * 100
significant_missing_all = missing_all[missing_all > 20].sort_values(ascending=False)

print("\n=== Significant Missing Values (>20%) - All Surgeries ===")
print(significant_missing_all.round(1).astype(str) + "%")

# === Step 4: Missing Values Analysis for Canceled Surgeries ===
missing_canceled = df_canceled.isnull().mean() * 100
significant_missing_canceled = missing_canceled[missing_canceled > 20].sort_values(ascending=False)

print("\n=== Significant Missing Values (>20%) - Canceled Surgeries ===")
print(significant_missing_canceled.round(1).astype(str) + "%")

# === Step 5: Cancellation rate (by patient ID) ===
df_all['was_canceled'] = df_all['ת"ז מותממת'].isin(df_canceled['ת"ז מותממת'])
overall_cancel_rate = df_all['was_canceled'].mean() * 100
print(f"\nOverall Cancellation Rate: {overall_cancel_rate:.2f}%")

# === Step 6: Cancellation Analysis by Age Group ===
df_all['age_group'] = pd.cut(df_all['גיל המטופל בזמן הניתוח'], 
                             bins=[0, 18, 35, 50, 65, 80, 120],
                             labels=['Child', 'Young Adult', 'Adult', 'Mid Age', 'Senior', 'Elderly'])
age_cancel_rate = df_all.groupby('age_group')['was_canceled'].mean() * 100
print("\n=== Cancellation Rate by Age Group ===")
print(age_cancel_rate.round(1).astype(str) + "%")

# === Step 7: Cancellation Analysis by Weekday ===
df_all['weekday'] = pd.to_datetime(df_all['תאריך ביצוע ניתוח']).dt.day_name()
weekday_cancel_rate = df_all.groupby('weekday')['was_canceled'].mean() * 100
print("\n=== Cancellation Rate by Day of Week ===")
print(weekday_cancel_rate.round(1).astype(str) + "%")

# === Step 8: Timing Statistics (Days from Request to Surgery) ===
timing_stats = df_all['ימים מתאריך פתיחת בקשה ועד תאריך ביצוע ניתוח'].describe().round(1)
print("\n=== Days from Request to Surgery (Statistics) ===")
print(timing_stats[['mean', 'std', 'min', 'max']])
