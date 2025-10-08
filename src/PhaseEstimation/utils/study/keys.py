import pickle
import re
from pathlib import Path

import numpy as np


def get_keys(path, label=""):
    # Build regex depending on whether a label is given
    if label:
        pattern = re.compile(
            rf"{re.escape(label)}_k(-?\d+(?:\.\d+)?)_h(-?\d+(?:\.\d+)?)\.pkl$"
        )
    else:
        pattern = re.compile(r"k(-?\d+(?:\.\d+)?)_h(-?\d+(?:\.\d+)?)\.pkl$")

    kh_list = []
    for file in path.glob("*.pkl"):
        match = pattern.match(file.name)
        if match:
            k, h = map(float, match.groups())
            kh_list.append((k, h))
    return kh_list


def dir_to_img(path, label, key=None):
    # unpack keys and values
    ks, hs = zip(*get_keys(path, label))

    # sort unique values
    ks = np.sort(np.unique(ks))
    hs = np.sort(np.unique(hs))

    img = np.zeros((len(hs), len(ks)))
    for x, k in enumerate(ks):
        for y, h in enumerate(hs):
            with open(Path(path) / f"{label}_k{k}_h{h}.pkl", "rb") as f:
                if key is None:
                    img[x, y] = pickle.load(f)
                else:
                    img[x, y] = pickle.load(f)[key]

    return img, ks, hs
