import pickle
import re
from pathlib import Path


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


def load_dict(path, label=""):
    input_dict = {}
    kh = get_keys(Path(path), label)
    for k, h in kh:
        filename = (
            Path(path) / f"{label}_k{k}_h{h}.pkl"
            if label
            else Path(path) / f"k{k}_h{h}.pkl"
        )
        with open(filename, "rb") as f:
            input_dict[(k, h)] = pickle.load(f)

    return input_dict
