"""
Module to generate MPS for the ANNNI model using DMRG and save them to a folder.
The module takes as input the number of spins L, the discretization of the phase space side, the maximum bond dimension chi and the path where to save the MPS.
The module generates all the MPS for the ANNNI model with the given parameters and saves them to the given path.

"""

import argparse 
import os

parser = argparse.ArgumentParser()
parser.add_argument("-L", type=int, default=8, help="Number of spins")
parser.add_argument("-side", type=int, default=51, help="Discretization of the phase space")
parser.add_argument("-chi", type=int, default=8, help="Maximum bond dimension")
parser.add_argument("-path", type=str, default="./mps/", help="Path to save the MPS")

args = parser.parse_args()

if not os.path.exists(args.path):
    os.makedirs(args.path)

script_path = "/home/samonaco/Quantum-Phase-Detection-ANNNI/src/PhaseEstimation/dmrg.py"
    
filename = f"{args.path}ANNNI_L{args.L}_X{args.chi}.pkl"
if os.path.exists(filename):
    print(f"Skipping L={args.L}, chi={args.chi}, file already exists.")
else:
    os.system(f"python {script_path} --L {args.L} --side {args.side} --chi {args.chi} --path '{args.path}'")