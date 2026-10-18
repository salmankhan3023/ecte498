import pandas as pd
import numpy as np
import os
import json
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f

# ============================================================
# 1. SETUP & LOAD SAVED ASSETS
# ============================================================
SAVE_DIR = r"D:\Subjects\ECTE 498\IDS\Salman scripts\Final\Thesis run"
OUT_DIR = os.path.join(SAVE_DIR, "Test results")
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 0.96
SEED = 42
LAT_SAMPLE = 10000       # packets used for the one-at-a-time latency run
CPU_POWER_W = 25.0       # assumed avg CPU power under load (not measured), E = P * t
MAX_FLOW_PAYLOAD = 4096
ENTROPY_SAMPLE = 0.30

print("--- Loading Models and Test Data ---")
with open(os.path.join(SAVE_DIR, 'features.json')) as f:
    feats = json.load(f)
t1_cols, t2_cols, t3_cols = feats['t1_cols'], feats['t2_cols'], feats['t3_cols']

le = joblib.load(os.path.join(SAVE_DIR, 'label_encoder.joblib'))
rf_t1 = joblib.load(os.path.join(SAVE_DIR, 'tier1_rf.joblib'))
rf_t2 = joblib.load(os.path.join(SAVE_DIR, 'tier2_rf.joblib'))
scaler_t3 = joblib.load(os.path.join(SAVE_DIR, 'scaler_t3.joblib'))
dl_model = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'tier3_dl_model.keras'))

df_test = pd.read_csv(os.path.join(SAVE_DIR, 'test.csv'))
y_true = le.transform(df_test['Label'])
class_names = le.classes_
benign_idx = le.transform(['BENIGN'])[0]
n_test = len(df_test)
print(f"✅ {n_test:,} test samples, {len(class_names)} classes")

# t3_cols = t2_cols + extras and t2_cols = t1_cols + entropy, so one matrix serves all tiers
X3 = df_test[t3_cols].values.astype(np.float32)
X2 = X3[:, :len(t2_cols)]
X1 = X3[:, :len(t1_cols)]

# ============================================================
# 2. TIERED INFERENCE (full test set)
# ============================================================
print("\n--- Running Tiered Inference ---")
final_preds = np.zeros(n_test, dtype=int)
tier_log = np.empty(n_test, dtype=object)
analyst_alert = np.zeros(n_test, dtype=bool)

p1 = rf_t1.predict_proba(X1)
t1_mask = p1.max(axis=1) >= THRESHOLD
final_preds[t1_mask] = rf_t1.classes_[p1[t1_mask].argmax(axis=1)]
tier_log[t1_mask] = 'Tier 1'

rem_idx = np.where(~t1_mask)[0]
if len(rem_idx):
    p2 = rf_t2.predict_proba(X2[rem_idx])
    t2_local = p2.max(axis=1) >= THRESHOLD
    t2_idx = rem_idx[t2_local]
    final_preds[t2_idx] = rf_t2.classes_[p2[t2_local].argmax(axis=1)]
    tier_log[t2_idx] = 'Tier 2'

    t3_idx = rem_idx[~t2_local]
    if len(t3_idx):
        p3 = dl_model.predict(scaler_t3.transform(X3[t3_idx]), batch_size=4096, verbose=0)
        final_preds[t3_idx] = p3.argmax(axis=1)
        tier_log[t3_idx] = 'Tier 3'
        analyst_alert[t3_idx] = p3.max(axis=1) < THRESHOLD   # sent to manual review

tiers = ['Tier 1', 'Tier 2', 'Tier 3']
tier_colors = {'Tier 1': 'steelblue', 'Tier 2': 'seagreen', 'Tier 3': 'mediumpurple'}
for t in tiers:
    print(f"   {t}: {(tier_log == t).sum():,} ({(tier_log == t).mean()*100:.2f}%)")
print(f"   Tier 3 flows flagged for analyst review: {analyst_alert.sum():,}")

# ============================================================
# 3. VISUALIZATIONS & SAVING
# ============================================================
print("\n--- Generating Plots ---")
labels_all = np.arange(len(class_names))

cm = confusion_matrix(y_true, final_preds, labels=labels_all)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Final Tiered System Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'))
plt.close()

# normalised confusion matrix (recall per class)
cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', xticklabels=class_names, yticklabels=class_names)
plt.title('Normalised Confusion Matrix (Row = True Class)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix_normalised.png'))
plt.close()

# tier distribution
tier_counts = [(tier_log == t).sum() for t in tiers]
plt.figure(figsize=(8, 6))
plt.bar(tiers, tier_counts, color=[tier_colors[t] for t in tiers])
plt.title('Traffic Distribution Across Tiers')
plt.ylabel('Packet Count')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tier_distribution.png'))
plt.close()

# text report
report = classification_report(y_true, final_preds, labels=labels_all, target_names=class_names, digits=4, zero_division=0)
with open(os.path.join(OUT_DIR, "final_performance.txt"), "w", encoding="utf-8") as f:
    f.write(report)

# per-class table
report_dict = classification_report(y_true, final_preds, labels=labels_all, target_names=class_names,
                                    digits=4, output_dict=True, zero_division=0)
metrics_rows = []
for i, cls in enumerate(class_names):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - tp - fn - fp
    is_attack = cls != 'BENIGN'
    metrics_rows.append({
        'Class': cls, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Precision': report_dict[cls]['precision'],
        'Recall': report_dict[cls]['recall'],
        'F1': report_dict[cls]['f1-score'],
        'Support': report_dict[cls]['support'],
        'Missed Attacks (predicted BENIGN)': cm[i, benign_idx] if is_attack else 0,
        'False Flags (BENIGN→Class)': cm[benign_idx, i] if is_attack else 0
    })
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(os.path.join(OUT_DIR, 'per_class_metrics.csv'), index=False)

# missed attacks & false flags
attack_df = metrics_df[metrics_df['Class'] != 'BENIGN']
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(attack_df['Class'], attack_df['Missed Attacks (predicted BENIGN)'], color='crimson')
axes[0].set_title('Missed Attacks per Class\n(attack predicted as BENIGN)')
axes[0].set_xlabel('Count')
axes[1].barh(attack_df['Class'], attack_df['False Flags (BENIGN→Class)'], color='darkorange')
axes[1].set_title('False Flags per Attack Class\n(BENIGN misclassified as Attack)')
axes[1].set_xlabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'missed_attacks_false_flags.png'))
plt.close()

# per-tier accuracy
tier_results = []
for t in tiers:
    m = tier_log == t
    if m.sum() == 0:
        continue
    tier_results.append({'Tier': t, 'Samples': int(m.sum()), 'Accuracy': accuracy_score(y_true[m], final_preds[m])})
tier_df = pd.DataFrame(tier_results)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(tier_df['Tier'], tier_df['Accuracy'], color=[tier_colors[t] for t in tier_df['Tier']])
ax.set_ylim(0, 1.05)
ax.set_ylabel('Accuracy')
ax.set_title('Per-Tier Classification Accuracy')
for bar, row in zip(bars, tier_df.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{row.Accuracy:.4f}\n(n={row.Samples:,})', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'per_tier_accuracy.png'))
plt.close()

# class distribution per tier
tcc_df = pd.DataFrame({t: np.bincount(y_true[tier_log == t], minlength=len(class_names)) for t in tiers},
                      index=class_names)
tcc_df.plot(kind='bar', stacked=True, figsize=(14, 6), colormap='tab20')
plt.title('Class Distribution Per Tier (True Labels)')
plt.ylabel('Sample Count')
plt.xlabel('Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tier_class_distribution.png'))
plt.close()

# precision / recall / f1
metrics_df[['Class', 'Precision', 'Recall', 'F1']].set_index('Class').plot(kind='bar', figsize=(16, 6), colormap='Set2')
plt.title('Per-Class Precision, Recall & F1')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'per_class_prf1.png'))
plt.close()

# summary
overall_acc = accuracy_score(y_true, final_preds)
attack_mask = y_true != benign_idx
detection_rate = (final_preds[attack_mask] != benign_idx).mean()
fpr_overall = (final_preds[~attack_mask] != benign_idx).mean()
total_missed = int((attack_mask & (final_preds == benign_idx)).sum())
total_false_flags = int((~attack_mask & (final_preds != benign_idx)).sum())

summary_lines = [
    "=" * 50,
    "EXTENDED SUMMARY METRICS",
    "=" * 50,
    f"Test samples              : {n_test:,}  ({attack_mask.sum():,} attacks)",
    f"Overall Accuracy          : {overall_acc:.4f}",
    f"Attack Detection Rate     : {detection_rate:.4f}  (attacks not predicted as BENIGN)",
    f"False Positive Rate       : {fpr_overall:.4f}  (benign flagged as attack)",
    f"Total Missed Attacks (FN) : {total_missed}",
    f"Total False Flags  (FP)   : {total_false_flags}",
    f"Tier 3 analyst alerts     : {int(analyst_alert.sum())}",
    "",
    "Tier Breakdown:",
]
for row in tier_df.itertuples():
    summary_lines.append(f"  {row.Tier}: {row.Samples:,} samples ({row.Samples/n_test*100:.3f}%), Accuracy={row.Accuracy:.4f}")
with open(os.path.join(OUT_DIR, "extended_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print("\n".join(summary_lines))

# ============================================================
# 4. LATENCY & ENERGY (one packet at a time)
# ============================================================
print("\n--- Latency & Energy Simulation ---")

@njit(cache=True)
def shannon_entropy(payload, ratio):
    counts = np.zeros(256, dtype=np.int64)
    n = 0
    for b in payload:
        if np.random.random() < ratio:
            counts[b] += 1
            n += 1
    if n == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            h -= p * np.log2(p)
    return h

def dl_infer(x):
    return dl_model(x, training=False).numpy()

# single-row forest calls are dominated by thread dispatch with n_jobs=-1
rf_t1.n_jobs = 1
rf_t2.n_jobs = 1

rng = np.random.default_rng(SEED)
n_lat = min(LAT_SAMPLE, n_test)
samp = rng.choice(n_test, n_lat, replace=False)

# the csv only stores the entropy value, so a payload of the recorded size is generated
# (outside the timer) to measure what the Numba entropy step costs per flow
pay_len = np.minimum(df_test['total_payload_bytes'].values[samp], MAX_FLOW_PAYLOAD).astype(int)
payloads = [rng.integers(0, 256, size=max(n, 0), dtype=np.uint8) for n in pay_len]

# warm-up so JIT / graph tracing isn't counted
shannon_entropy(payloads[0], ENTROPY_SAMPLE)
for _ in range(5):
    rf_t1.predict_proba(X1[:1]); rf_t2.predict_proba(X2[:1]); dl_infer(scaler_t3.transform(X3[:1]))

# --- tiered system ---
t_lat, t_tier, t_pred = [], [], []
for k, i in enumerate(samp):
    t0 = time.perf_counter()
    p = rf_t1.predict_proba(X1[i:i+1])[0]
    tier, pred = 1, rf_t1.classes_[p.argmax()]
    if p.max() < THRESHOLD:
        shannon_entropy(payloads[k], ENTROPY_SAMPLE)
        p = rf_t2.predict_proba(X2[i:i+1])[0]
        tier, pred = 2, rf_t2.classes_[p.argmax()]
        if p.max() < THRESHOLD:
            p = dl_infer(scaler_t3.transform(X3[i:i+1]))[0]
            tier, pred = 3, p.argmax()
    t_lat.append((time.perf_counter() - t0) * 1000)
    t_tier.append(tier)
    t_pred.append(pred)
    if (k + 1) % 2000 == 0:
        print(f"   Tiered: {k+1:,} / {n_lat:,}")

# --- pure deep learning on the same packets ---
d_lat, d_pred = [], []
for k, i in enumerate(samp):
    t0 = time.perf_counter()
    p = dl_infer(scaler_t3.transform(X3[i:i+1]))[0]
    d_lat.append((time.perf_counter() - t0) * 1000)
    d_pred.append(p.argmax())
    if (k + 1) % 2000 == 0:
        print(f"   Pure DL: {k+1:,} / {n_lat:,}")

t_lat, d_lat = np.array(t_lat), np.array(d_lat)
t_tier, t_pred, d_pred = np.array(t_tier), np.array(t_pred), np.array(d_pred)
t_energy = t_lat * CPU_POWER_W      # ms * W = mJ
d_energy = d_lat * CPU_POWER_W

y_s = y_true[samp]
tier_acc_s = accuracy_score(y_s, t_pred)
dl_acc_s = accuracy_score(y_s, d_pred)
agree = (t_pred == final_preds[samp]).mean()

tier_lat = {t: t_lat[t_tier == t] for t in (1, 2, 3)}
tier_share = {t: (t_tier == t).mean() * 100 for t in (1, 2, 3)}
speedup = d_lat.mean() / t_lat.mean()
lat_reduction = (1 - t_lat.mean() / d_lat.mean()) * 100
energy_saving = (1 - t_energy.sum() / d_energy.sum()) * 100

# Fig: latency per tier + weighted avg
fig, ax = plt.subplots(figsize=(9, 6))
avg_by_tier = [tier_lat[t].mean() if len(tier_lat[t]) else 0 for t in (1, 2, 3)]
bars = ax.bar(tiers, avg_by_tier, color=[tier_colors[t] for t in tiers])
ax.axhline(t_lat.mean(), color='green', linestyle='--', label=f'Weighted avg: {t_lat.mean():.2f} ms')
for bar, t in zip(bars, (1, 2, 3)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{bar.get_height():.2f} ms\n{tier_share[t]:.2f}% of traffic', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('Latency (ms)')
ax.set_title('Average End-to-End Latency per Tier')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'latency_per_tier.png'))
plt.close()

# Fig: latency distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].hist(t_lat, bins=60, alpha=0.6, color='green', label='Tiered')
axes[0].hist(d_lat, bins=60, alpha=0.6, color='red', label='Pure DL')
axes[0].axvline(t_lat.mean(), color='green', linestyle='--')
axes[0].axvline(d_lat.mean(), color='red', linestyle='--')
axes[0].set_xlabel('Latency (ms)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Latency Distribution: Tiered vs Pure DL')
axes[0].legend()
present = [t for t in (1, 2, 3) if len(tier_lat[t])]
axes[1].boxplot([tier_lat[t] for t in present] + [d_lat], showmeans=True)
axes[1].set_xticklabels([f'Tier {t}' for t in present] + ['Pure DL'])
axes[1].set_ylabel('Latency (ms)')
axes[1].set_title('Latency by Tier')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'latency_distribution.png'))
plt.close()

# Fig: cumulative energy
plt.figure(figsize=(9, 6))
cum_t, cum_d = np.cumsum(t_energy), np.cumsum(d_energy)
plt.plot(cum_t, color='green', label='Tiered System')
plt.plot(cum_d, color='red', label='Pure DL')
plt.fill_between(range(n_lat), cum_t, cum_d, color='green', alpha=0.3, label='Energy Saved')
plt.xlabel('Packet Number')
plt.ylabel('Cumulative Energy (mJ)')
plt.title('Cumulative Energy Consumption')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cumulative_energy.png'))
plt.close()

# Fig: avg energy per packet
plt.figure(figsize=(7, 6))
vals = [t_energy.mean(), d_energy.mean()]
b = plt.bar(['Tiered System', 'Pure DL'], vals, color=['green', 'red'], alpha=0.7)
for bar, v in zip(b, vals):
    plt.text(bar.get_x() + bar.get_width()/2, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
plt.ylabel('Energy (mJ)')
plt.title('Average Energy Consumption per Packet')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'avg_energy_per_packet.png'))
plt.close()

pd.DataFrame({
    'True_Label': le.inverse_transform(y_s),
    'Tiered_Pred': le.inverse_transform(t_pred.astype(int)),
    'DL_Pred': le.inverse_transform(d_pred.astype(int)),
    'Tier': t_tier, 'Tiered_Latency_ms': t_lat, 'DL_Latency_ms': d_lat,
    'Tiered_Energy_mJ': t_energy, 'DL_Energy_mJ': d_energy
}).to_csv(os.path.join(OUT_DIR, 'latency_energy_results.csv'), index=False)

lat_lines = [
    "=" * 60,
    "LATENCY & ENERGY REPORT",
    "=" * 60,
    f"Packets timed (one at a time) : {n_lat:,}",
    f"Assumed CPU power             : {CPU_POWER_W} W  (energy = power x latency)",
    "",
    "Metric                  Pure DL        Tiered",
    f"Avg latency (ms)        {d_lat.mean():<14.3f} {t_lat.mean():.3f}",
    f"Median latency (ms)     {np.median(d_lat):<14.3f} {np.median(t_lat):.3f}",
    f"Throughput (pkts/s)     {1000/d_lat.mean():<14.1f} {1000/t_lat.mean():.1f}",
    f"Avg energy (mJ/pkt)     {d_energy.mean():<14.3f} {t_energy.mean():.3f}",
    f"Accuracy on sample (%)  {dl_acc_s*100:<14.2f} {tier_acc_s*100:.2f}",
    "",
    f"Latency reduction : {lat_reduction:.2f}%  ({speedup:.2f}x)",
    f"Energy saving     : {energy_saving:.2f}%",
    f"Sequential vs batch prediction agreement: {agree*100:.2f}%",
    "",
    "Per tier (end-to-end, includes earlier tiers):",
]
for t in (1, 2, 3):
    if len(tier_lat[t]):
        lat_lines.append(f"  Tier {t}: {len(tier_lat[t]):,} packets ({tier_share[t]:.2f}%), avg {tier_lat[t].mean():.3f} ms")
    else:
        lat_lines.append(f"  Tier {t}: no packets in sample")
with open(os.path.join(OUT_DIR, "latency_energy_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lat_lines))
print("\n".join(lat_lines))

print(f"\n--- TESTING COMPLETE. All results saved in {OUT_DIR} ---")
