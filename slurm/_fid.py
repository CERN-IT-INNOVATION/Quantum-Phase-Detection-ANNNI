import argparse
import pickle
from pathlib import Path

import numpy as np
import pennylane as qml
import quimb as qu


def fidelity(wf1, wf2):
    wf1 = wf1 / np.linalg.norm(wf1)
    wf2 = wf2 / np.linalg.norm(wf2)

    inner_product = np.vdot(wf1, wf2)
    return np.abs(inner_product) ** 2


def main():
    parser = argparse.ArgumentParser(description="Main script for getting states")

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
        "--depth",
        type=int,
        required=True,
    )

    parser.add_argument("--path", type=str, required=True)

    args = parser.parse_args()

    dev = qml.device("default.qubit", wires=args.spins)

    file_dmrg = Path(args.path) / "data" / f"dmrg_k{args.k}_h{args.h}.pkl"
    file_brick = Path(args.path) / "data" / f"brick{args.depth}_k{args.k}_h{args.h}.pkl"
    file_diag = Path(args.path) / "data" / f"diag_k{args.k}_h{args.h}.pkl"

    output_path = Path(args.path) / f"brick{args.depth}" / "fid"

    with open(file_dmrg, "rb") as f:
        mps_raw = qu.tensor.tensor_1d.MatrixProductState(pickle.load(f), shape="lpr")
        psi_dmrg = mps_raw.to_dense()

    with open(file_brick, "rb") as f:
        pp_wire, pp_U = pickle.load(f)

        @qml.qnode(dev)
        def ansatz_state():
            for p_wire, p_U in zip(pp_wire[-args.depth :], pp_U[-args.depth :]):
                for wire, U in zip(p_wire, p_U):
                    qml.QubitUnitary(U, wires=wire)
            return qml.state()

        psi_brick = ansatz_state()

    with open(file_diag, "rb") as f:
        psi_diag = pickle.load(f)

    fid_dmrg_diag_fw = output_path / f"fid_dmrg_diag_k{args.k}_h{args.h}.pkl"
    fid_dmrg_brick_fw = output_path / f"fid_dmrg_brick_k{args.k}_h{args.h}.pkl"
    fid_brick_diag_fw = output_path / f"fid_brick_diag_k{args.k}_h{args.h}.pkl"

    with open(fid_dmrg_diag_fw, "wb") as f:
        pickle.dump(fidelity(psi_dmrg, psi_diag), f)

    with open(fid_dmrg_brick_fw, "wb") as f:
        pickle.dump(fidelity(psi_dmrg, psi_brick), f)

    with open(fid_brick_diag_fw, "wb") as f:
        pickle.dump(fidelity(psi_brick, psi_diag), f)


if __name__ == "__main__":
    main()
