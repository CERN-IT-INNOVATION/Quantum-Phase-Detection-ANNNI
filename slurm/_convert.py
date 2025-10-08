import argparse
from pathlib import Path

from PhaseEstimation.inputs import Brick

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument("--h", type=float, required=True)

    args = parser.parse_args()

    file_dmrg = Path(args.path) / f"dmrg_k{args.k}_h{args.h}.pkl"

    with open(file_dmrg, "rb") as f:
        mps = pickle.load(f)

    Brick.make(args.k, args.h, mps, args.depth, args.path, label=f"brick{args.depth}")
