""" This module implements the base function for treating Ising Chain with Transverse Field. """
import pennylane as qml
from pennylane import numpy as np

##############

def get_H(N, lam, J):
    """
    Set up Hamiltonian:
            H = -lam*Σsigma^i_z - J*Σsigma^i_x*sigma_x^{i+1}

    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    lam : float
        Strenght of (transverse) magnetic field
    J : float
        Interaction strenght between spins

    Returns
    -------
    pennylane.ops.qubit.hamiltonian.Hamiltonian
        Hamiltonian Pennylane class for the (Transverse) Ising Chain
    """
    # Interaction of spins with magnetic field
    H = -lam * qml.PauliZ(0)
    for i in range(1, N):
        H = H - lam * qml.PauliZ(i)

    # Interaction between spins:
    for i in range(0, N - 1):
        H = H + J * (-1) * (qml.PauliX(i) @ qml.PauliX(i + 1))

    return H

def build_Hs(N, J, n_states):
    """
    Sets up np.ndarray of pennylane Hamiltonians with different instensity of magnetic field mu in np.linspace(0, 2*J, n_states)
    
    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    J : float
        Interaction strenght between spins
    n_states : int
        Number of Hamiltonians to generate
    """
    lams = np.linspace(0, 2*J, n_states)
    
    Hs     = []
    labels = []
    for lam in lams:
        Hs.append(get_H(int(N), float(lam), float(J)) )
        labels.append(0) if lam <= J else labels.append(1)
        
    return Hs, labels

