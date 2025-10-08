import argparse
import os

import matplotlib.pyplot as plt

from PhaseEstimation.utils.study import RESULT_PATH
from PhaseEstimation.utils.study.dirs import make_dir
from PhaseEstimation.utils.study.keys import dir_to_img, get_keys
from PhaseEstimation.utils.study.plot import imgs
from PhaseEstimation.utils.study.scripts import make_script, run_script


def main():
    parser = argparse.ArgumentParser(description="Main script for getting states")

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
        help="Number of iterations of the brick-layered ansatz (default: 10)",
    )

    parser.add_argument(
        "--plot", action="store_true", help="Plot data if set to True (default: False)"
    )

    args = parser.parse_args()

    print("Master Script:")
    print(f"num_spins: {args.spins}")
    print(f"side:      {args.side}")
    print(f"dmrg_chi:  {args.chi}")

    main_path = RESULT_PATH / f"L{args.spins}_chi{args.chi}_side{args.side}"
    data_path = main_path / "data"
    plot_path = main_path / f"brick{args.depth}" / "plot"
    make_dir(plot_path)
    fid_path = main_path / f"brick{args.depth}" / "fid"
    make_dir(fid_path)
    script_path = main_path / "scripts"
    make_dir(script_path)

    kh_dmrg = get_keys(data_path, "dmrg")
    kh_diag = get_keys(data_path, "diag")
    kh_brick = get_keys(data_path, f"brick{args.depth}")

    # DMRG - DIAG check keys
    if set(kh_dmrg) != set(kh_diag):
        print("DMRG-EXACT")
        print("Keys don't match")
        print(f"DMRG ({len(kh_dmrg)})")
        print(f"EXACT ({len(kh_diag)})")

    # DIAG - BRICK check keys
    if set(kh_diag) != set(kh_brick):
        print("DIAG-BRICK")
        print("Keys don't match")
        print(f"DIAG  ({len(kh_diag)})")
        print(f"BRICK ({len(kh_brick)})")

    if not args.plot:
        for k, h in kh_dmrg:
            filenames = [
                fid_path / f"{appendix}_k{k}_h{h}"
                for appendix in [
                    "fid_dmrg_diag",
                    "fid_brick_diag",
                    "fid_dmrg_brick",
                ]
            ]
            python_line = f"""
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/slurm/_fid.py \\
--spins {args.spins} \\
--depth {args.depth} \\
--k {k} \\
--h {h} \\
--path {main_path.resolve()} \\
"""
            if any(not os.path.exists(filename) for filename in filenames):
                make_script(
                    script_path,
                    f"FID{args.spins}_{args.depth}_k{k}_h{h}",
                    "FID",
                    python_line,
                )
                run_script(script_path, f"FID{args.spins}_{args.depth}_k{k}_h{h}")

    else:
        dmrgbrick_fid, _, _ = dir_to_img(fid_path, "fid_dmrg_brick")
        brickdiag_fid, _, _ = dir_to_img(fid_path, "fid_brick_diag")
        diagdmrg_fid, _, _ = dir_to_img(fid_path, "fid_dmrg_diag")

        imgs(
            [dmrgbrick_fid, brickdiag_fid, diagdmrg_fid],
            [f"DMRG-BRICK({args.depth})", f"BRICK({args.depth})-DIAG", "DIAG-DMRG"],
            f"ANNNI L={args.spins}: Fidelities",
            [0, 1],
        )
        plt.savefig(plot_path / "Fidelities.pdf")
        plt.close()


if __name__ == "__main__":
    main()
