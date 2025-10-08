import argparse
from pathlib import Path

import h5py

from PhaseEstimation.inputs import Brick, Diag, Mps
from PhaseEstimation.qcnn.model import Qcnn
from PhaseEstimation.qcnn.plot import loss_curve, predict_all, predict_single
from PhaseEstimation.qcnn.train import train

parser = argparse.ArgumentParser()
parser.add_argument("--ansatz", type=str, required=True)
parser.add_argument("--numepochs", type=int, required=True)
parser.add_argument("--lr", type=float, required=True)
parser.add_argument("--T", type=float, required=True)
parser.add_argument("--datapath", type=str, required=True)
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--bs", type=int, required=True)
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

qcnn = Qcnn(Input, args.ansatz)
loss_value, params, loss_history, params_history = train(
    qcnn, num_epochs=args.numepochs, lr=args.lr, T=args.T, path=args.path, showbar=False
)
loss_curve(loss_history, T=args.T, path=args.path)

predictions = qcnn.predict(params, sequential=True)
predict_single(predictions, qcnn.num_qubits, T=args.T, path=args.path)
predict_all(predictions, num_qubits=qcnn.num_qubits, T=args.T, path=args.path)

with h5py.File(Path(args.path) / "data.h5", "w") as f:
    f.create_dataset("params", data=params.tolist())
    f.create_dataset("predictions", data=predictions)
