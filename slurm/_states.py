import argparse
import pickle
from pathlib import Path

import pennylane as qml
import quimb as qu

from PhaseEstimation.inputs import Brick, Diag, Mps
from PhaseEstimation.utils import fidelity

parser = argparse.ArgumentParser(description="DMRG states of the ANNNI model")

parser.add_argument(
    "--spins",
    type=int,
    required=True,
)

parser.add_argument(
    "--k",
    type=float,
    required=True,
)

parser.add_argument(
    "--h",
    type=float,
    required=True,
)

parser.add_argument(
    "--chi",
    type=int,
    required=True,
)

parser.add_argument(
    "--path",
    type=str,
    required=True,
)

parser.add_argument(
    "--diag",
    type=bool,
    required=True,
)

parser.add_argument(
    "--fid",
    type=bool,
    required=True,
)

parser.add_argument(
    "--depth",
    type=int,
    required=True,
)


args = parser.parse_args()

if args.diag:
    Diag.make(args.k, args.h, args.spins, args.path, "diag")

Mps.make(args.k, args.h, args.spins, args.chi, args.path, "dmrg")


file_dmrg = Path(args.path) / f"dmrg_k{args.k}_h{args.h}.pkl"

with open(file_dmrg, "rb") as f:
    mps = pickle.load(f)

Brick.make(args.k, args.h, mps, args.depth, args.path, label=f"brick{args.depth}")

if args.fid:
    dev = qml.device("default.qubit", wires=args.spins)

    file_dmrg = Path(args.path) / f"dmrg_k{args.k}_h{args.h}.pkl"
    file_brick = Path(args.path) / f"brick{args.depth}_k{args.k}_h{args.h}.pkl"

    output_path = Path(args.path).parent / "fid"

    with open(file_dmrg, "rb") as f:
        mps_raw = qu.tensor.tensor_1d.MatrixProductState(pickle.load(f), shape="lpr")
        psi_dmrg = mps_raw.to_dense()

    with open(file_brick, "rb") as f:
        pp_wire, pp_U = pickle.load(f)

    for depth in range(args.depth):

        @qml.qnode(dev)
        def ansatz_state():
            for p_wire, p_U in zip(
                pp_wire[-(args.depth - depth) :], pp_U[-(args.depth - depth) :]
            ):
                for wire, U in zip(p_wire, p_U):
                    qml.QubitUnitary(U, wires=wire)
            return qml.state()

        psi_brick = ansatz_state()

        fid_dmrg_brick_fw = (
            output_path / f"fid_dmrg_brick{depth+1}_k{args.k}_h{args.h}.pkl"
        )

        with open(fid_dmrg_brick_fw, "wb") as f:
            pickle.dump(fidelity(psi_dmrg, psi_brick), f)
