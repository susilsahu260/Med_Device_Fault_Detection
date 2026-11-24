import pandas as pd
import numpy as np

df = pd.read_csv('synthetic_ventilator_equipment_dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Check step indices within each device
df['step_idx'] = df.groupby('device_id').cumcount()

print("=== FAULT DISTRIBUTION BY STEP INDEX ===")
# See which step indices have failure == 1
step_failure_rate = df.groupby('step_idx')['failure'].mean()
print("Step indices with failure > 0:")
print(step_failure_rate[step_failure_rate > 0])

# Crosstab of step_idx (or binned) vs fault_type
step_faults = pd.crosstab(df['step_idx'], df['fault_type'])
print("\nNon-zero fault types across step indices:")
print(step_faults[step_faults['normal'] < 100].head(25))

# Check whether faults are injected at fixed intervals or random blocks
print("\nFault episode lengths per device:")
episode_lengths = []
for dev, group in df.groupby('device_id'):
    # compute run-length encoding of failure
    failures = group['failure'].values
    faults = group['fault_type'].values
    # find blocks
    changes = np.diff(np.concatenate(([0], failures, [0])))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for s, e in zip(starts, ends):
        fault_name = group['fault_type'].iloc[s]
        episode_lengths.append({
            'device_id': dev,
            'start_step': s,
            'end_step': e,
            'length': e - s,
            'fault_type': fault_name
        })

episodes_df = pd.DataFrame(episode_lengths)
print(f"Total fault episodes: {len(episodes_df)}")
print(episodes_df.groupby('fault_type')['length'].describe())
print("\nEpisode start step distribution by fault type:")
print(episodes_df.groupby('fault_type')['start_step'].describe())

# Check correlations with failure
print("\n=== CORRELATION OF NUMERIC FEATURES WITH 'failure' ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_failure = df[numeric_cols].corr()['failure'].sort_values(ascending=False)
print(corr_failure)
