import jax.numpy as jnp
import numpy as np
import pennylane as qml
from jax import jit

j_linalgeigh = jit(jnp.linalg.eigh)


def get_H(num_spins, k, h, ring=False):
    """Construction function the ANNNI Hamiltonian (J=1)"""

    # Interaction between spins (neighbouring):
    H = -1 * (qml.PauliX(0) @ qml.PauliX(1))
    for i in range(1, num_spins - 1):
        H = H - (qml.PauliX(i) @ qml.PauliX(i + 1))

    # Interaction between spins (next-neighbouring):
    for i in range(0, num_spins - 2):
        H = H + k * (qml.PauliX(i) @ qml.PauliX(i + 2))

    # Interaction of the spins with the magnetic field
    for i in range(0, num_spins):
        H = H - h * qml.PauliZ(i)

    if ring:
        if num_spins > 2:
            # Nearest-neighbour wrap-around
            H = H - (qml.PauliX(num_spins - 1) @ qml.PauliX(0))
        # Next-nearest-neighbour wrap-around only if num_spins > 4
        # (for 4 spins, (0,2) and (1,3) already exist)
        if num_spins > 4:
            H = H + k * (qml.PauliX(num_spins - 2) @ qml.PauliX(0))
            H = H + k * (qml.PauliX(num_spins - 1) @ qml.PauliX(1))

    return H


def get_energy_psi(num_spins, k, h, ring=False, usejax=False):
    # Get the matrix form of the Hamiltonian and ensure it's real
    mat_H = np.real(qml.matrix(get_H(num_spins, k, h, ring))).astype(np.single)

    if usejax:
        # Compute sorted eigenvalues and eigenvectors using jitted function
        eigvals, psi = j_linalgeigh(mat_H)  # psi contains eigenvectors as columns
    else:
        eigvals, psi = np.linalg.eigh(mat_H)

    # Get the ground-state energy (smallest eigenvalue)
    en0 = eigvals[0]

    # Get the ground-state wavefunction (first eigenvector)
    psi0 = psi[:, 0]

    return en0, psi0
