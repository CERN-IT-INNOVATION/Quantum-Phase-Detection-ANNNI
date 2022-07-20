""" This module implements the base function for treating the Ising Chain with Nearest Interactions and Next-Nearest Interactions without Magnetic Field. """
import pennylane as qml
from pennylane import numpy as np

##############

def get_H(N, j2, J1 = 1, ring = False):
    """
    Set up Hamiltonian:
    .. math:: H = -J1 (*\\Sigma \\sigma_x^i*\\sigma_x^{i+1}] + (J2/J1)*\\Sigma \\sigma^i_x*\\sigma_x^{i+2})

    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    j2 : float
        Strenght of next-neighbour interactions
    J1 : float
        Interaction strenght between nearest spins
    ring : bool
        If False, system has open-boundaries condition

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
        
    # If ring == True, the 'chain' needs to be closed
    if ring:
        # Next nearest interactions:
        H = H + (j2/J1) * (qml.PauliX(N - 2) @ qml.PauliX(0))
        H = H + (j2/J1) * (qml.PauliX(N - 1) @ qml.PauliX(1))
        
    H = (-J1) * H

    return H

def build_Hs(N, n_states, J1 = 1,  ring = False):
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

    Returns
    -------
    np.array
        Array of pennylane Hamiltonians
    np.array
        Array of labels for analytical solutionss
    np.array
        Array for the recycle rule
    np.array
        Array for the states parameters
    """
    # Array of free parameters (next-nearest interaction strenght)
    js = np.linspace(0, -1*J1, n_states)
    
    Hs     = []       # Array of pennylane hamiltonians
    labels = []       # Array of labels:
                      #  > 0 if J2 <= -J1/2
                      #  > 1 if J2 >  -J/2
    ising_params = [] # Array of parameters [N,J1,J2]
    
    for j in js:
        Hs.append(get_H(int(N), float(j), float(J1), ring) )
        labels.append(1) if j<= -J1/2 else labels.append(0)
        ising_params.append([N, J1, j])
      
    # Array of indices for the order of states to train through VQE
    recycle_rule = np.arange(n_states)
    
    return Hs, np.array(labels), np.array(recycle_rule), np.array(ising_params)

