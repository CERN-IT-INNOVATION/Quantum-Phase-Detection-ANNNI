import argparse

from PhaseEstimation.inputs import Mps

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


args = parser.parse_args()
Mps.make(args.k, args.h, args.spins, args.chi, args.path, "dmrg")
