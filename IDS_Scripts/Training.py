import pandas as pd
import numpy as np
import os
import json
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input

# ============================================================
# 1. SETUP
# ============================================================
SAVE_DIR = r"D:\Subjects\ECTE 498\IDS\Salman scripts\Final\Thesis run"

SEED = 42
RF_TREES = 100
RF_MAX_DEPTH = None     
CLASS_CAP = 50000        # BENIGN and any other huge class are cut down to this
SMOTE_TARGET = 5000      # minority classes are boosted to this
MIN_FOR_SMOTE = 50       # below this SMOTE has too few neighbours to be meaningful
EPOCHS = 10
BATCH_SIZE = 2048

tf.keras.utils.set_random_seed(SEED)

with open(os.path.join(SAVE_DIR, 'features.json')) as f:
    feats = json.load(f)
t1_cols, t2_cols, t3_cols = feats['t1_cols'], feats['t2_cols'], feats['t3_cols']

# ============================================================
# 2. DATA LOADING
# ============================================================
print("--- Loading Training Data ---")
train_df = pd.read_csv(os.path.join(SAVE_DIR, 'train.csv'))
print(f"✅ {len(train_df):,} training rows")

le = LabelEncoder()
y_train = le.fit_transform(train_df['Label'])
joblib.dump(le, os.path.join(SAVE_DIR, 'label_encoder.joblib'))
print(f"✅ {len(le.classes_)} classes: {', '.join(le.classes_)}")

# ============================================================
# 3. TRAINING TIER 1 & 2 (Random Forest)
# ============================================================
print("\n--- Training Tier 1 (metadata only) ---")
t0 = time.time()
rf_t1 = RandomForestClassifier(n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH, n_jobs=-1, random_state=SEED)
rf_t1.fit(train_df[t1_cols].values, y_train)
t_rf1 = time.time() - t0
print(f"✅ Tier 1 done in {t_rf1:.1f}s")

print("--- Training Tier 2 (metadata + entropy) ---")
t0 = time.time()
rf_t2 = RandomForestClassifier(n_estimators=RF_TREES, max_depth=RF_MAX_DEPTH, n_jobs=-1, random_state=SEED)
rf_t2.fit(train_df[t2_cols].values, y_train)
t_rf2 = time.time() - t0
print(f"✅ Tier 2 done in {t_rf2:.1f}s")

joblib.dump(rf_t1, os.path.join(SAVE_DIR, 'tier1_rf.joblib'))
joblib.dump(rf_t2, os.path.join(SAVE_DIR, 'tier2_rf.joblib'))

# ============================================================
# 4. TRAINING TIER 3 (Undersample + SMOTE + MLP)
# ============================================================
print("\n--- Balancing Tier 3 ---")
scaler_t3 = StandardScaler()
X_t3 = scaler_t3.fit_transform(train_df[t3_cols].values)
joblib.dump(scaler_t3, os.path.join(SAVE_DIR, 'scaler_t3.joblib'))

counts = Counter(y_train)
cap_strategy = {c: CLASS_CAP for c, n in counts.items() if n > CLASS_CAP}
X_u, y_u = RandomUnderSampler(sampling_strategy=cap_strategy, random_state=SEED).fit_resample(X_t3, y_train)

counts_u = Counter(y_u)
smote_strategy = {c: SMOTE_TARGET for c, n in counts_u.items() if MIN_FOR_SMOTE <= n < SMOTE_TARGET}
skipped = [le.classes_[c] for c, n in counts_u.items() if n < MIN_FOR_SMOTE]
X_res, y_res = SMOTE(sampling_strategy=smote_strategy, random_state=SEED).fit_resample(X_u, y_u)

print(f"   After undersampling: {len(X_u):,} | After SMOTE: {len(X_res):,}")
print(f"   Too small for SMOTE (left as is): {skipped}")

dl_model = Sequential([
    Input(shape=(len(t3_cols),)),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])
dl_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n--- Training Tier 3 MLP ---")
t0 = time.time()
dl_model.fit(X_res, y_res, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
t_dl = time.time() - t0
dl_model.save(os.path.join(SAVE_DIR, 'tier3_dl_model.keras'))
print(f"✅ Tier 3 done in {t_dl:.1f}s")

# ============================================================
# 5. SAVE TRAINING SUMMARY
# ============================================================
summary = {
    'train_rows': int(len(train_df)),
    'rf_trees': RF_TREES, 'rf_max_depth': RF_MAX_DEPTH,
    'class_cap': CLASS_CAP, 'smote_target': SMOTE_TARGET, 'min_for_smote': MIN_FOR_SMOTE,
    'tier3_train_rows_after_resampling': int(len(X_res)),
    'tier3_class_counts': {le.classes_[c]: int(n) for c, n in sorted(Counter(y_res).items())},
    'train_time_s': {'tier1': t_rf1, 'tier2': t_rf2, 'tier3': t_dl}
}
with open(os.path.join(SAVE_DIR, 'training_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n--- TRAINING COMPLETE. Models saved in {SAVE_DIR} ---")
