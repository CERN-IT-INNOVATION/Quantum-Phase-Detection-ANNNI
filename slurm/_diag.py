import argparse

from PhaseEstimation.inputs import Diag

parser = argparse.ArgumentParser(description="DIAG states of the ANNNI model")

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
Diag.make(args.k, args.h, args.spins, args.path, "diag")
