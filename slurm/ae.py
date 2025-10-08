import argparse
import json
import os

from PhaseEstimation.utils.study import RESULT_PATH
from PhaseEstimation.utils.study.dirs import make_dir
from PhaseEstimation.utils.study.scripts import make_script, run_script


def get_study_name(args):
    return f"A{args.ansatz}_d{args.depth}_nt{args.numtrash}_lr{args.lr}_ne{args.numepochs}_k{args.k}_h{args.h}_R{args.R}"


def get_study_json(args):
    return {
        "spins": args.spins,
        "side": args.side,
        "chi": args.chi,
        "Udepth": args.depth,
        "numtrials": args.numtrials,
        "depth": args.depth,
        "ansatz": args.ansatz,
        "numtrash": args.numtrash,
        "lr": args.lr,
        "numepochs": args.numepochs,
        "k": args.k,
        "h": args.h,
        "R": args.R,
    }


def main():
    parser = argparse.ArgumentParser(description="Main script for training Autoencoder")

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
        "--numtrials",
        type=int,
        default=100,
        help="Number of trainings (default: 100)",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Depth of the autoencoder (default: 2)",
    )

    parser.add_argument(
        "--ansatz",
        type=str,
        default="anomaly",
        help="Ansatz name (default: anomaly)",
    )

    parser.add_argument(
        "--numtrash",
        type=int,
        default=3,
        help="Number of epochs (default: 100)",
    )

    parser.add_argument(
        "--numepochs",
        type=int,
        default=500,
        help="Number of epochs (default: 100)",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )

    parser.add_argument(
        "--label",
        type=str,
        default="diag",
        help="Label of the input",
    )

    parser.add_argument(
        "--k",
        type=float,
        default=0,
        help="k",
    )

    parser.add_argument(
        "--h",
        type=float,
        default=0,
        help="h",
    )

    parser.add_argument(
        "--R",
        type=int,
        default=0,
        help="Radius",
    )

    args = parser.parse_args()
    study_name = get_study_name(args)

    print("Master Script:")
    print(f"num_spins: {args.spins}")
    print(f"side:      {args.side}")
    print(f"dmrg_chi:  {args.chi}")

    main_path = RESULT_PATH / f"L{args.spins}_chi{args.chi}_side{args.side}"
    data_path = main_path / "data"

    study_path = main_path / "ae" / study_name / args.label
    make_dir(study_path)
    script_path = main_path / "scripts"
    make_dir(script_path)

    with open(study_path / "study.json", "w") as f:
        json.dump(get_study_json(args), f, indent=4)

    for it in range(args.numtrials):
        substudy_path = study_path / f"{it}"
        make_dir(substudy_path)
        filename = study_path / f"{it}" / "data.h5"
        python_line = f"""
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/slurm/_ae.py \\
--depth {args.depth} \\
--ansatz {args.ansatz} \\
--numepochs {args.numepochs} \\
--lr {args.lr} \\
--numtrash {args.numtrash} \\
--datapath {data_path.resolve()} \\
--path {substudy_path.resolve()} \\
--label {args.label} \\
--k {args.k} \\
--h {args.h} \\
--R {args.R}
"""
        if not os.path.exists(filename):
            make_script(script_path, f"AE{study_name}_{it}", "AE", python_line)
            run_script(script_path, f"AE{study_name}_{it}")


if __name__ == "__main__":
    main()
