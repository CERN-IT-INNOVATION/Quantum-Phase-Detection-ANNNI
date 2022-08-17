""" This module implements the base function for treating Ising Chain with Transverse Field. """
import pennylane as qml
from pennylane import numpy as np

from typing import List, Tuple
from numbers import Number
##############

def get_H(N : int, lam : float, J : float, ring : bool = False) -> qml.ops.qubit.hamiltonian.Hamiltonian:
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
    ring : bool
        If False, system has open-boundaries condition

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
        
    if ring:
        # Set interaction between first and last spin
        H = H + J * (-1) * (qml.PauliX(N-1) @ qml.PauliX(0) )

    return H

def build_Hs(N : int, J : float, n_states : int, ring : bool = False) -> Tuple[List[qml.ops.qubit.hamiltonian.Hamiltonian], List[int], List[int], List[Tuple[int, Number, Number]], int]:
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
        
    Returns
    -------
    np.array
        Array of pennylane Hamiltonians
    np.array
        Array of labels for analytical solutions
    np.array
        Array for the recycle rule
    np.array
        Array for the states parameters
    int
        Number of states
    """
    # Array of free parameters (magnetic field)
    lams = np.linspace(0, 2*J, n_states)
    
    Hs     = []       # Array of pennylane hamiltonians
    labels = []       # Array of labels:
                      #  > 0 if magnetic field <= J
                      #  > 1 if magnetic field >  J 
    ising_params = [] # Array of parameters [N,J,magneticfield]
    
    for lam in lams:
        Hs.append(get_H(int(N), float(lam), float(J), ring) )
        labels.append(0) if lam <= J else labels.append(1)
        ising_params.append([N, J, lam])
    
    # Array of indices for the order of states to train through VQE
    recycle_rule = np.arange(n_states)
    
    return Hs, np.array(labels), np.array(recycle_rule), np.array(ising_params), n_states

