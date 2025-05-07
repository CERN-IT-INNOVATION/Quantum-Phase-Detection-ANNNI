# %%

import argparse 
import os
import numpy as np
import jax.numpy as jnp
import pennylane as qml
from jax import vmap
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-L", type=int, default=8, help="Number of spins")
parser.add_argument("-side", type=int, default=51, help="Discretization of the phase space")
parser.add_argument("-path", type=str, default="./mps/", help="Path to save the MPS")

args = parser.parse_args()
# %%
if not os.path.exists(args.path):
    os.makedirs(args.path)

# %%
    
filename = f"{args.path}ANNNI_L{args.L}.pkl"
if os.path.exists(filename):
    print(f"Skipping L={args.L}, file already exists.")
else:
    # Create meshgrid of the parameter space
    def diagonalize_H(H_matrix):
        """Returns the lowest eigenvector of the Hamiltonian matrix."""
        _, psi = jnp.linalg.eigh(H_matrix)  # Compute eigenvalues and eigenvectors
        return jnp.array(psi[:, 0], dtype=jnp.complex64)  # Return the ground state

    def get_H(num_spins, k, h):
        """Construction function the ANNNI Hamiltonian (J=1)"""

        # Interaction between spins (neighbouring):
        H = -1 * (qml.PauliX(0) @ qml.PauliX(1))
        for i in range(1, num_spins - 1):
            H = H  - (qml.PauliX(i) @ qml.PauliX(i + 1))

        # Interaction between spins (next-neighbouring):
        for i in range(0, num_spins - 2):
            H = H + k * (qml.PauliX(i) @ qml.PauliX(i + 2))

        # Interaction of the spins with the magnetic field
        for i in range(0, num_spins):
            H = H - h * qml.PauliZ(i)

        return H

    p_h = np.linspace(0.0, 2.0, args.side)
    p_k = np.linspace(0.0, 1.0, args.side)

    # Preallocate arrays for Hamiltonian matrices and phase labels.
    H_matrices = np.empty((len(p_k), len(p_h), 2**args.L, 2**args.L))
    phases = np.empty((len(p_k), len(p_h)), dtype=int)

    for x, k in enumerate(p_k):
        for y, h in enumerate(p_h):
            H_matrices[y, x] = np.real(qml.matrix(get_H(args.L, k, h))) # Get Hamiltonian matrix

    # Vectorized diagonalization
    pp_psi = vmap(vmap(diagonalize_H))(H_matrices)

    # Create the dictionary from the vmapped outputs
    psi_dict = {
        (float(p_k[x]), float(p_h[y])): pp_psi[y, x]
        for y in range(len(p_h))
        for x in range(len(p_k))
    }

    # Save entire dictionary as a single pickle file
    save_path = f"{args.path}/ANNNI_L{args.L}.pkl"
    with open(save_path, "wb") as file:
        pickle.dump(psi_dict, file)
