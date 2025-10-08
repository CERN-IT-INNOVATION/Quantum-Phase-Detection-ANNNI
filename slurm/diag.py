import argparse
import os

from PhaseEstimation.inputs.utils import get_keys
from PhaseEstimation.utils.study import RESULT_PATH
from PhaseEstimation.utils.study.dirs import make_dir
from PhaseEstimation.utils.study.scripts import make_script, run_script


def main():
    parser = argparse.ArgumentParser(
        description="Main script for getting states DIAGONALIZATION"
    )

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

    args = parser.parse_args()

    print("Master Script:")
    print(f"num_spins: {args.spins}")
    print(f"side:      {args.side}")
    print(f"chi:       {args.chi}")

    # Get the unique name of the experiment from the arguments
    data_path = RESULT_PATH / f"L{args.spins}_chi{args.chi}_side{args.side}" / "data"

    script_path = (
        RESULT_PATH / f"L{args.spins}_chi{args.chi}_side{args.side}" / "scripts"
    )
    make_dir(script_path)

    p_kh = get_keys(data_path, "dmrg")

    for k, h in p_kh:
        filename = data_path / f"diag_k{k}_h{h}.pkl"
        python_line = f"""
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/slurm/_diag.py \\
--spins {args.spins} \\
--chi {args.chi} \\
--k {k} \\
--h {h} \\
--path {data_path.resolve()}
"""
        if not os.path.exists(filename):
            make_script(script_path, f"DIAG_k{k}_h{h}", "DIAG", python_line)
            run_script(script_path, f"DIAG_k{k}_h{h}")


if __name__ == "__main__":
    main()
