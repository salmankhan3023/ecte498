import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

# ============================================================
# 1. SETUP & FEATURE DEFINITIONS
# ============================================================
BASE_DIR = r"D:\Subjects\ECTE 498\IDS\Salman scripts"
file_paths = [os.path.join(BASE_DIR, f"merged_{d}_with_entropy.csv")
              for d in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']]

SAVE_DIR = os.path.join(BASE_DIR, "Final", "Thesis run")
os.makedirs(SAVE_DIR, exist_ok=True)

TEST_SIZE = 0.2
SEED = 42

metadata_features = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Bwd Packet Length Max', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd Packets/s', 'Bwd Packets/s', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'PSH Flag Count', 'ACK Flag Count', 'Fwd Header Length'
]
entropy_col = 'entropy_mean'
tier3_extra = [
    'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
    'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'Bwd Header Length', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'total_payload_bytes'
]

t1_cols, t2_cols = metadata_features, metadata_features + [entropy_col]
t3_cols = t2_cols + tier3_extra

# ============================================================
# 2. DATA LOADING
# ============================================================
print("--- Loading Data ---")
df_list = []
for p in file_paths:
    if not os.path.exists(p):
        print(f"!!! Missing file: {p}")
        continue
    d = pd.read_csv(p, low_memory=False, encoding_errors='replace')
    d.columns = d.columns.str.strip()
    print(f"   {os.path.basename(p)}: {len(d):,} rows")
    df_list.append(d[t3_cols + ['Label']])

df = pd.concat(df_list, axis=0, ignore_index=True)
del df_list
print(f"✅ Loaded {len(df):,} rows")

# ============================================================
# 3. CLEANING
# ============================================================
print("\n--- Cleaning ---")
df[t3_cols] = df[t3_cols].apply(pd.to_numeric, errors='coerce')

# inf -> max finite value of that column (-inf -> min)
for c in t3_cols:
    finite = df[c].replace([np.inf, -np.inf], np.nan)
    df[c] = df[c].replace(np.inf, finite.max()).replace(-np.inf, finite.min())

before = len(df)
df = df.dropna(subset=t3_cols + ['Label'])
print(f"   NaN rows dropped: {before - len(df):,}")

# the web attack labels come in with a broken dash character
df['Label'] = (df['Label'].astype(str)
               .str.replace(r'^Web Attack\s*\W+\s*', 'Web Attack: ', regex=True)
               .str.strip())

# duplicates are removed BEFORE the split so the same flow can't land in both sets
before = len(df)
df = df.drop_duplicates(subset=t3_cols + ['Label']).reset_index(drop=True)
print(f"   Duplicate rows removed: {before - len(df):,}")

df[t3_cols] = df[t3_cols].astype(np.float32)
print(f"✅ Clean dataset: {len(df):,} rows")
print(df['Label'].value_counts().to_string())

# ============================================================
# 4. STRATIFIED TRAIN / TEST SPLIT
# ============================================================
print("\n--- Splitting ---")
train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df['Label'], random_state=SEED)

train_df.to_csv(os.path.join(SAVE_DIR, 'train.csv'), index=False)
test_df.to_csv(os.path.join(SAVE_DIR, 'test.csv'), index=False)

with open(os.path.join(SAVE_DIR, 'features.json'), 'w') as f:
    json.dump({'t1_cols': t1_cols, 't2_cols': t2_cols, 't3_cols': t3_cols}, f, indent=2)

dist = pd.DataFrame({'Train': train_df['Label'].value_counts(),
                     'Test': test_df['Label'].value_counts()}).fillna(0).astype(int)
dist.to_csv(os.path.join(SAVE_DIR, 'class_distribution.csv'))

print(f"✅ Train: {len(train_df):,} | Test: {len(test_df):,}")
print(dist.to_string())
print(f"\n--- DONE. Files saved in {SAVE_DIR} ---")
