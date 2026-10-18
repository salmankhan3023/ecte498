import dpkt
import socket
import numpy as np
import pandas as pd
import os
import glob
from collections import defaultdict
from entropy_utils import shannon_entropy, seed_entropy, MAX_FLOW_PAYLOAD, SAMPLE_RATIO

# ============================================================
# 1. SETUP
# ============================================================
PCAP_DIR = r"D:\Subjects\ECTE 498\IDS\CICIDS2017\PCAPs"
CSV_DIR = r"D:\Subjects\ECTE 498\IDS\CICIDS2017\TrafficLabelling"    
OUT_DIR = r"D:\Subjects\ECTE 498\IDS\Salman scripts"
SEED = 42

days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
match_cols = ['Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol']

os.makedirs(OUT_DIR, exist_ok=True)
seed_entropy(SEED)


# ============================================================
# 2. FLOW MATCHING (bidirectional 5-tuple)
# ============================================================
def make_key(ip_a, port_a, ip_b, port_b, proto):
    a, b = (ip_a, port_a), (ip_b, port_b)
    return (a, b, proto) if a <= b else (b, a, proto)


def build_flow_map(df):
    flow_map = defaultdict(list)
    for i, (sip, sp, dip, dp, pr) in enumerate(df[match_cols].itertuples(index=False, name=None)):
        try:
            key = make_key(socket.inet_aton(sip), int(sp), socket.inet_aton(dip), int(dp), int(pr))
        except (OSError, ValueError, TypeError):
            continue
        flow_map[key].append(i)
    return flow_map


# ============================================================
# 3. PCAP PASS (first 4096 payload bytes per flow)
# ============================================================
def read_payloads(pcap_path, flow_map):
    bufs = {}
    n_pkts = n_hit = 0
    with open(pcap_path, "rb") as f:
        try:
            pcap = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            pcap = dpkt.pcapng.Reader(f)

        for ts, buf in pcap:
            n_pkts += 1
            try:
                ip = dpkt.ethernet.Ethernet(buf).data
                if not isinstance(ip, dpkt.ip.IP) or ip.p not in (6, 17):
                    continue
                l4 = ip.data
                if not isinstance(l4, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    continue
                key = make_key(ip.src, l4.sport, ip.dst, l4.dport, ip.p)
                if key not in flow_map:
                    continue
                n_hit += 1
                b = bufs.setdefault(key, bytearray())
                if len(b) < MAX_FLOW_PAYLOAD and len(l4.data) > 0:
                    b.extend(l4.data[:MAX_FLOW_PAYLOAD - len(b)])
            except Exception:
                continue

            if n_pkts % 2000000 == 0:
                print(f"   {n_pkts:,} packets read...")

    print(f"   Packets: {n_pkts:,} | matched to a flow: {n_hit:,}")
    return bufs


# ============================================================
# 4. MAIN LOOP
# ============================================================
report_rows = []

for day in days:
    print("\n" + "=" * 60)
    print(f"{day.upper()}")
    print("=" * 60)

    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, f"{day.capitalize()}-*.csv")))
    pcap_files = glob.glob(os.path.join(PCAP_DIR, f"{day.capitalize()}-*.pcap"))
    if not csv_files or not pcap_files:
        print(f"!!! Missing csv or pcap for {day}, skipping")
        continue

    df = pd.concat([pd.read_csv(p, encoding='latin-1', low_memory=False) for p in csv_files], ignore_index=True)
    df.columns = df.columns.str.strip()
    n_rows = len(df)
    df['flow_uid'] = [f"{day}|{i}" for i in range(n_rows)]
    print(f"✅ {n_rows:,} flows from {len(csv_files)} csv file(s)")

    flow_map = build_flow_map(df)
    print(f"   {len(flow_map):,} unique 5-tuples")

    print("Reading pcap...")
    bufs = read_payloads(pcap_files[0], flow_map)

    # one entropy value per 5-tuple, shared by every csv flow that uses it
    ent_by_key = {k: (shannon_entropy(np.frombuffer(b, dtype=np.uint8), SAMPLE_RATIO) if len(b) else 0.0)
                  for k, b in bufs.items()}

    entropy = np.zeros(n_rows, dtype=np.float32)
    payload_len = np.zeros(n_rows, dtype=np.int32)
    in_pcap = np.zeros(n_rows, dtype=bool)
    chunks = [b''] * n_rows
    for key, rows in flow_map.items():
        b = bufs.get(key)
        if b is None:
            continue
        data = bytes(b)
        for i in rows:
            entropy[i] = ent_by_key[key]
            payload_len[i] = len(data)
            in_pcap[i] = True
            chunks[i] = data

    df['entropy_mean'] = entropy
    df['total_payload_bytes'] = payload_len
    df.to_csv(os.path.join(OUT_DIR, f"merged_{day}_with_entropy.csv"), index=False)

    # raw payload store (flat byte array + offsets) so entropy can be recomputed live at test time
    offsets = np.concatenate([[0], np.cumsum([len(c) for c in chunks])]).astype(np.int64)
    np.save(os.path.join(OUT_DIR, f"payloads_{day}_data.npy"), np.frombuffer(b''.join(chunks), dtype=np.uint8))
    np.save(os.path.join(OUT_DIR, f"payloads_{day}_offsets.npy"), offsets)

    # matching report
    rep = pd.DataFrame({'Label': df['Label'], 'in_pcap': in_pcap, 'has_payload': payload_len > 0})
    rep = rep.groupby('Label').agg(flows=('in_pcap', 'size'), matched=('in_pcap', 'mean'), with_payload=('has_payload', 'mean'))
    rep.insert(0, 'Day', day)
    report_rows.append(rep.reset_index())
    print(f"✅ Matched to pcap: {in_pcap.mean()*100:.2f}% | with payload: {(payload_len > 0).mean()*100:.2f}%")
    print(rep.round(3).to_string())

if report_rows:
    pd.concat(report_rows).to_csv(os.path.join(OUT_DIR, 'entropy_match_report.csv'), index=False)
print("\n--- DONE ---")
