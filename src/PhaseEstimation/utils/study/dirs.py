from pathlib import Path


def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
