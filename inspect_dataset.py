import pandas as pd
import numpy as np
import json

df = pd.read_csv('synthetic_ventilator_equipment_dataset.csv')

print("=== SHAPE ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== COLUMNS & DTYPES ===")
print(df.dtypes)

print("\n=== HEAD (3 rows) ===")
print(df.head(3).T)

print("\n=== MISSING VALUES ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_Pct': missing_pct})
print(missing_df[missing_df['Missing_Count'] > 0] if (missing > 0).any() else "No missing values found.")

print("\n=== DUPLICATE ROWS ===")
duplicates = df.duplicated().sum()
print(f"Total duplicate rows: {duplicates}")

print("\n=== UNIQUE VALUES PER COLUMN ===")
for col in df.columns:
    n_unique = df[col].nunique()
    print(f"{col}: {n_unique} unique values ({df[col].dtype})")
    if n_unique <= 20:
        val_counts = df[col].value_counts(dropna=False).to_dict()
        print(f"   Values: {val_counts}")

print("\n=== POTENTIAL TARGETS / LABELS ===")
# Look for categorical columns, failure/status/fault/condition/target/label
potential_targets = [col for col in df.columns if df[col].dtype == 'object' or df[col].nunique() <= 20]
for pt in potential_targets:
    print(f"\nValue distribution for '{pt}':")
    vc = df[pt].value_counts(dropna=False)
    vcp = df[pt].value_counts(normalize=True, dropna=False) * 100
    dist_df = pd.DataFrame({'Count': vc, 'Percentage (%)': vcp})
    print(dist_df)

print("\n=== TIME / SEQUENCE INVESTIGATION ===")
time_cols = [c for c in df.columns if any(t in c.lower() for t in ['time', 'date', 'timestamp', 'step', 'epoch', 'sec'])]
print(f"Potential time columns: {time_cols}")

for tc in time_cols:
    print(f"\nExamining column: {tc}")
    print(f"Sample values: {df[tc].head(5).tolist()}")
    # Check if convertible to datetime or numeric
    try:
        dt_series = pd.to_datetime(df[tc])
        print(f"Successfully parsed as datetime. Min: {dt_series.min()}, Max: {dt_series.max()}")
        is_sorted = dt_series.is_monotonic_increasing
        print(f"Is strictly monotonically increasing: {is_sorted}")
        time_diffs = dt_series.diff()
        print(f"Time differences summary:\n{time_diffs.describe()}")
        print(f"Top 5 most frequent time steps:\n{time_diffs.value_counts().head(5)}")
    except Exception as e:
        print(f"Not standard datetime or failed to parse: {e}")
        if np.issubdtype(df[tc].dtype, np.number):
            print(f"Numeric sequence. Monotonic increasing: {df[tc].is_monotonic_increasing}")
            diffs = df[tc].diff()
            print(f"Numeric differences summary:\n{diffs.describe()}")

# Check for ID columns
id_cols = [c for c in df.columns if any(i in c.lower() for i in ['id', 'patient', 'device', 'unit', 'breath', 'session'])]
print(f"\nPotential ID/Group columns: {id_cols}")
for ic in id_cols:
    print(f"{ic}: {df[ic].nunique()} unique identifiers. Top values: {df[ic].value_counts().head(5).to_dict()}")

print("\n=== NUMERIC COLUMNS SUMMARY ===")
num_cols = df.select_dtypes(include=[np.number]).columns
print(df[num_cols].describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']])
