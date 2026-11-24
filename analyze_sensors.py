import pandas as pd
import numpy as np

df = pd.read_csv('synthetic_ventilator_equipment_dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("=== OPERATING HOURS CHECK ===")
op_inc = [group['operating_hours'].is_monotonic_increasing for _, group in df.groupby('device_id')]
print(f"Operating hours strictly monotonically increasing per device: {all(op_inc)} ({sum(op_inc)}/100)")
# check delta of operating hours per 5-min step
op_diffs = [group['operating_hours'].diff().dropna() for _, group in df.groupby('device_id')]
all_op_diffs = pd.concat(op_diffs)
print(f"Operating hours delta summary (5 min step in hours = 5/60 = 0.08333h):")
print(all_op_diffs.describe())
print(f"Most frequent deltas:\n{all_op_diffs.round(3).value_counts().head(5)}")

print("\n=== DEVICE MODEL CONSISTENCY ===")
models_per_dev = df.groupby('device_id')['device_model'].nunique()
print(f"Max models per device: {models_per_dev.max()}, Min models per device: {models_per_dev.min()}")

# Device model count of unique devices
dev_model_counts = df.groupby('device_model')['device_id'].nunique()
print("Unique devices per model:")
print(dev_model_counts)

print("\n=== FEATURE DISTRIBUTIONS & OUTLIER SCREENING ===")
sensor_cols = [
    'inlet_pressure_kPa', 'airway_pressure_cmH2O', 'airflow_L_min',
    'oxygen_concentration_pct', 'internal_temperature_C', 'supply_voltage_V',
    'motor_current_A', 'blower_speed_rpm', 'battery_voltage_V',
    'vibration_mm_s', 'filter_pressure_drop_kPa', 'ambient_humidity_pct'
]

stats_list = []
for col in sensor_cols:
    q25 = df[col].quantile(0.25)
    q75 = df[col].quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    stats_list.append({
        'feature': col,
        'mean': df[col].mean(),
        'std': df[col].std(),
        'min': df[col].min(),
        'p25': q25,
        'median': df[col].median(),
        'p75': q75,
        'max': df[col].max(),
        'iqr_outliers_count': outliers,
        'iqr_outliers_pct': (outliers / len(df)) * 100
    })

stats_df = pd.DataFrame(stats_list)
print(stats_df.round(3).to_string())

print("\n=== SENSOR SIGNATURES PER FAULT TYPE ===")
# Detailed mean and std per fault type
for ft in df['fault_type'].unique():
    print(f"\n--- Fault Type: {ft} ---")
    sub = df[df['fault_type'] == ft]
    sub_mean = sub[sensor_cols].mean()
    normal_mean = df[df['fault_type'] == 'normal'][sensor_cols].mean()
    diff = sub_mean - normal_mean
    z_diff = diff / df[sensor_cols].std()
    sig = pd.DataFrame({'Fault_Mean': sub_mean, 'Normal_Mean': normal_mean, 'Diff': diff, 'Z_Shift': z_diff})
    # show features with significant shift
    print(sig.sort_values(by='Z_Shift', key=abs, ascending=False).head(5).round(3))
