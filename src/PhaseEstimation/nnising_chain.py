""" This module implements the base function for treating the Ising Chain with Nearest Interactions and Next-Nearest Interactions without Fields. """
import pennylane as qml
from pennylane import numpy as np

##############

def get_H(N, j2, J1 = 1):
    """
    Set up Hamiltonian:
            H = -J1 (*Σsigma_x^i*sigma_x^{i+1] + (J2/J1)*Σsigma^i_x*sigma_x^{i+2})

    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    j2 : float
        Strenght of next-neighbour interactions
    J1 : float
        Interaction strenght between nearest spins

    Returns
    -------
    pennylane.ops.qubit.hamiltonian.Hamiltonian
        Hamiltonian Pennylane class for the (Transverse) Ising Chain
    """
    # Interaction of nearest spins
    H = ( qml.PauliX(0) @ qml.PauliX(1) )
    for i in range(1, N - 1):
        H = H + ( qml.PauliX(i) @ qml.PauliX(i + 1) )

    # Interaction between next-neighbouring spins:
    for i in range(0, N - 2):
        H = H + (j2/J1) * (qml.PauliX(i) @ qml.PauliX(i + 2))
        
    H = (-J1) * H

    return H

def build_Hs(N, n_states, J1 = 1):
    """
    Sets up np.ndarray of pennylane Hamiltonians with strenghts of next-neighbours j2 in np.linspace(0, 1*J, n_states)
    
    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    n_states : int
        Number of Hamiltonians to generate
    J1 : float
        Interaction strenght between nearest spins
    """
    js = np.linspace(0, 1*J1, n_states)
    
    Hs     = []
    labels = []
    ising_params = []
    
    for j in js:
        Hs.append(get_H(int(N), float(j), float(J1)) )
        labels.append(0) if j<= J1/2 else labels.append(1)
        ising_params.append([N, J1, j])
        
    recycle_rule = np.arange(n_states)
    
    return Hs, labels, recycle_rule, ising_params

