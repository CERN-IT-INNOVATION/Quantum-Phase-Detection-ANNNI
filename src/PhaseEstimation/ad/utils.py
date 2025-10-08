from itertools import product

import numpy as np


def get_khs_train(model, k: float, h: float, R: int):
    ks, hs = model.ks, model.hs

    # Find closest indices
    k_idx = int(np.argmin(np.abs(ks - k)))
    h_idx = int(np.argmin(np.abs(hs - h)))

    # Collect (k, h) pairs starting from the closest
    khs = [(float(ks[k_idx]), float(hs[h_idx]))]

    # Generate all 8 directions in a 2D grid (±r for both k and h)
    directions = list(product([-1, 0, 1], repeat=2))
    directions.remove((0, 0))  # exclude center

    for r in range(1, R + 1):
        for dk, dh in directions:
            i, j = k_idx + dk * r, h_idx + dh * r
            if 0 <= i < len(ks) and 0 <= j < len(hs):
                khs.append((float(ks[i]), float(hs[j])))

    return khs
