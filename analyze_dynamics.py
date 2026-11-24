import pandas as pd
import numpy as np

df = pd.read_csv('synthetic_ventilator_equipment_dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("=== CHECKING PER-DEVICE TEMPORAL STRUCTURE ===")
# Check record count per device
device_counts = df['device_id'].value_counts()
print(f"Device count summary: min={device_counts.min()}, max={device_counts.max()}, mean={device_counts.mean()}")

# Check monotonic ordering per device
monotonic_checks = []
delta_checks = []
for dev, group in df.groupby('device_id'):
    is_inc = group['timestamp'].is_monotonic_increasing
    monotonic_checks.append(is_inc)
    diffs = group['timestamp'].diff().dropna()
    all_5min = (diffs == pd.Timedelta(minutes=5)).all()
    delta_checks.append(all_5min)

print(f"All devices have monotonically increasing timestamps: {all(monotonic_checks)} ({sum(monotonic_checks)}/{len(monotonic_checks)})")
print(f"All devices have strictly 5-minute sampling intervals: {all(delta_checks)} ({sum(delta_checks)}/{len(delta_checks)})")

# Check time duration per device
sample_dev = df[df['device_id'] == 'VENT-0001']
print(f"Sample device VENT-0001 start: {sample_dev['timestamp'].min()} end: {sample_dev['timestamp'].max()}")
duration = sample_dev['timestamp'].max() - sample_dev['timestamp'].min()
print(f"Duration per device: {duration} (exactly 199 steps of 5 min = 16.58 hours)")

# Check if timestamps overlap across devices or are sequential
device_time_ranges = df.groupby('device_id').agg(
    start=('timestamp', 'min'),
    end=('timestamp', 'max'),
    model=('device_model', 'first'),
    has_failure=('failure', 'max'),
    faults=('fault_type', lambda x: list(x.unique()))
)
print("\nFirst 10 devices time ranges:")
print(device_time_ranges.head(10))

# Check failure behavior over time within devices
print("\n=== FAILURE DYNAMICS WITHIN DEVICES ===")
devices_with_failure = device_time_ranges[device_time_ranges['has_failure'] == 1]
print(f"Devices that experience failure: {len(devices_with_failure)} / {len(device_time_ranges)}")

devices_healthy = device_time_ranges[device_time_ranges['has_failure'] == 0]
print(f"Devices that are entirely normal/healthy: {len(devices_healthy)} / {len(device_time_ranges)}")

# Look at failure transitions
transitions = []
for dev, group in df.groupby('device_id'):
    failures = group['failure'].tolist()
    fault_types = group['fault_type'].tolist()
    # Find index of first failure
    if 1 in failures:
        first_fail_idx = failures.index(1)
        # Check if failure persists until the end
        persists = all(f == 1 for f in failures[first_fail_idx:])
        fault_name = [ft for ft in fault_types if ft != 'normal']
        unique_faults = set(fault_name)
        transitions.append({
            'device_id': dev,
            'first_fail_idx': first_fail_idx,
            'fail_count': sum(failures),
            'persists_to_end': persists,
            'unique_faults': list(unique_faults)
        })

trans_df = pd.DataFrame(transitions)
print(f"\nFailure transition summary across {len(trans_df)} failed devices:")
print(f"First failure time step index: min={trans_df['first_fail_idx'].min()}, max={trans_df['first_fail_idx'].max()}, mean={trans_df['first_fail_idx'].mean():.1f}, median={trans_df['first_fail_idx'].median():.1f}")
print(f"All failures persist until end of run: {trans_df['persists_to_end'].all()}")
print(f"Single fault type per failed run: {all(len(f) == 1 for f in trans_df['unique_faults'])}")

# Check autocorrelation of sensor readings within individual devices
print("\n=== AUTOCORRELATION OF CONTINUOUS SENSORS (Lag-1 and Lag-5) ===")
sensor_cols = [
    'inlet_pressure_kPa', 'airway_pressure_cmH2O', 'airflow_L_min',
    'oxygen_concentration_pct', 'internal_temperature_C', 'supply_voltage_V',
    'motor_current_A', 'blower_speed_rpm', 'battery_voltage_V',
    'vibration_mm_s', 'filter_pressure_drop_kPa', 'ambient_humidity_pct'
]

autocorr_results = []
for col in sensor_cols:
    lag1_corrs = [group[col].autocorr(lag=1) for dev, group in df.groupby('device_id')]
    lag5_corrs = [group[col].autocorr(lag=5) for dev, group in df.groupby('device_id')]
    autocorr_results.append({
        'sensor': col,
        'mean_autocorr_lag1': np.nanmean(lag1_corrs),
        'mean_autocorr_lag5': np.nanmean(lag5_corrs)
    })
print(pd.DataFrame(autocorr_results).to_string())

print("\n=== FAULT DISTRIBUTION BY DEVICE MODEL ===")
print(pd.crosstab(df['device_model'], df['fault_type'], margins=True))

print("\n=== SENSOR STATS BY FAULT TYPE (Means) ===")
print(df.groupby('fault_type')[sensor_cols].mean().T.to_string())
