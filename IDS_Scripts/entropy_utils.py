import numpy as np
import os
from numba import njit

MAX_FLOW_PAYLOAD = 4096    # bytes kept per flow
SAMPLE_RATIO = 0.30        # share of those bytes used for the entropy


@njit(cache=True)
def seed_entropy(s):
    np.random.seed(s)


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


class PayloadStore:
    """reads the per-flow payload bytes written by 0_append_entropy.py"""

    def __init__(self, store_dir, days):
        self.data, self.off = {}, {}
        for d in days:
            self.data[d] = np.load(os.path.join(store_dir, f"payloads_{d}_data.npy"), mmap_mode='r')
            self.off[d] = np.load(os.path.join(store_dir, f"payloads_{d}_offsets.npy"))

    def get(self, uid):
        day, row = uid.split('|')
        row = int(row)
        o = self.off[day]
        return np.array(self.data[day][o[row]:o[row + 1]], dtype=np.uint8)
