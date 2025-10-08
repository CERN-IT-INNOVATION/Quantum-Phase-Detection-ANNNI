import argparse
import os

import numpy as np

from PhaseEstimation.utils.study import RESULT_PATH
from PhaseEstimation.utils.study.dirs import make_dir
from PhaseEstimation.utils.study.scripts import make_script, run_script


def main():
    parser = argparse.ArgumentParser(description="Main script for getting states DMRG")

    parser.add_argument(
        "--spins", type=int, default=8, help="Number of spins (default: 8)"
    )

    parser.add_argument(
        "--side",
        type=int,
        default=10,
        help="Discretization of the phase space (default: 10)",
    )

    parser.add_argument(
        "--chi", type=int, default=8, help="DMRG bond dimension (default: 8)"
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=10,
        help="Depth of the brick ansatz",
    )

    parser.add_argument(
        "--diag",
        action="store_true",
        default=False,
        help="Get states from diagonalization (default: False)",
    )

    parser.add_argument(
        "--fid",
        action="store_true",
        default=False,
        help="Compute fidelities",
    )

    args = parser.parse_args()

    print("Master Script:")
    print(f"num_spins: {args.spins}")
    print(f"side:      {args.side}")
    print(f"chi:       {args.chi}")

    # Get the unique name of the experiment from the arguments
    dmrg_path = RESULT_PATH / f"L{args.spins}_chi{args.chi}_side{args.side}" / "data"
    make_dir(dmrg_path)
    fid_path = RESULT_PATH / f"L{args.spins}_chi{args.chi}_side{args.side}" / "fid"
    make_dir(fid_path)

    script_path = (
        RESULT_PATH / f"L{args.spins}_chi{args.chi}_side{args.side}" / "scripts"
    )
    make_dir(script_path)

    hs = np.linspace(0.0, 2.0, args.side)
    ks = np.linspace(0.0, 1.0, args.side)

    for k in ks:
        for h in hs:
            filename = fid_path / f"fid_dmrg_brick{args.depth}_k{k}_h{h}.pkl"
            python_line = f"""
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/slurm/_states.py \\
--spins {args.spins} \\
--chi {args.chi} \\
--k {k} \\
--h {h} \\
--path {dmrg_path.resolve()} \\
--depth {args.depth} \\
--diag {args.diag} --fid {args.fid}
"""
            if not os.path.exists(filename):
                make_script(
                    script_path, f"STATES{args.spins}_k{k}_h{h}", "STATES", python_line
                )
                run_script(script_path, f"STATES{args.spins}_k{k}_h{h}")


if __name__ == "__main__":
    main()
