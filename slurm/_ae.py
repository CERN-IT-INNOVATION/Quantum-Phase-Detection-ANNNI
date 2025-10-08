import argparse
from pathlib import Path

import h5py

from PhaseEstimation.ad.model import Ad
from PhaseEstimation.ad.plot import compress, loss_curve
from PhaseEstimation.ad.train import train
from PhaseEstimation.inputs import Brick, Diag, Mps

parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=int, required=True)
parser.add_argument("--ansatz", type=str, required=True)
parser.add_argument("--numepochs", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--numtrash", type=int, required=True)
parser.add_argument("--datapath", type=str, required=True)
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--k", type=float, required=True)
parser.add_argument("--h", type=float, required=True)
parser.add_argument("--R", type=int, required=True)
args = parser.parse_args()


if args.label == "diag":
    # case 1: diag
    inputclass = Diag
elif args.label == "dmrg":
    # case 2: dmrg
    inputclass = Mps
elif args.label.startswith("brick") and args.label[5:].isdigit():
    # case 3: brick + integer
    inputclass = Brick
else:
    # optional: unknown case
    raise ValueError(f"Unknown label: {args.label}")

Input = inputclass(args.datapath, args.label)

autoencoder = Ad(
    Input, num_trashwires=args.numtrash, depth=args.depth, ansatz=args.ansatz
)
loss_value, params, loss_history, params_history = train(
    autoencoder,
    num_epochs=args.numepochs,
    lr=args.lr,
    k=args.k,
    h=args.h,
    R=args.R,
    path=args.path,
    showbar=False,
)

loss_curve(loss_history, path=args.path)
compression = autoencoder.compress(params)
compress(compression, autoencoder.num_qubits, path=args.path)

with h5py.File(Path(args.path) / "data.h5", "w") as f:
    f.create_dataset("params", data=params.tolist())
    f.create_dataset("compression", data=compression)
