""" This module implements the base function for treating ANNNI Ising-model """
import pennylane as qml
from pennylane import numpy as np

##############

def get_H(N, L, K, ring = False):
    """
    Set up Hamiltonian:
            H = J1* (- Σsigma^i_x*sigma_x^{i+1} - (h/J1) * Σsigma^i_z - (J2/J1) * Σsigma^i_x*sigma_x^{i+2}
        
        [where J1 = 1, (h/J1) = Lambda(/L), (J2/J1) = K]

    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    L : float
        TODO
    K : float
        TODO
    ring : bool
        If False, system has open-boundaries condition

    Returns
    -------
    pennylane.ops.qubit.hamiltonian.Hamiltonian
        Hamiltonian Pennylane class for the (Transverse) Ising Chain
    """
    # Interaction of spins with magnetic field
    H = - L * qml.PauliZ(0)
    for i in range(1, N):
        H = H - L * qml.PauliZ(i)

    # Interaction between spins (neighbouring):
    for i in range(0, N - 1):
        H = H + (-1) * (qml.PauliX(i) @ qml.PauliX(i + 1))
        
    # Interaction between spins (next-neighbouring):
    for i in range(0, N - 2):
        H = H + (-1) * K * (qml.PauliX(i) @ qml.PauliX(i + 2))
        
    # If ring == True, the 'chain' needs to be closed
    if ring:
        # Nearest interaction between last spin and first spin -particles
        H = H + (-1) * (qml.PauliX(N - 1) @ qml.PauliX(0))
        # Next nearest interactions:
        H = H + (-1) * K * (qml.PauliX(N - 2) @ qml.PauliX(0))
        H = H + (-1) * K * (qml.PauliX(N - 1) @ qml.PauliX(1))

    return H

def build_Hs(N, n_states, ring = False):
    """
    Sets up np.ndarray of pennylane Hamiltonians with different parameters
    total_states = n_states * n_states
    Taking n_states values of K from 0 to -1 (NB the sign)
    Taking n_states values of L from 0 to +2
    
    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    n_states : int
        Number of Hamiltonians to generate
    ring : bool
        If False, system has open-boundaries condition
        

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
    
    # Set up arrays of the parameters K and Ls
    K_states = np.linspace(0, -1, n_states)
    L_states = np.linspace(0, 2, n_states)
    
    Hs = []            # Array of the Pennylane hamiltonians
    labels = []        # Array of the labels:
                       #   > [1,1] for paramagnetic states
                       #   > [0,1] for ferromagnetic states
                       #   > [1,0] for antiphase states
                       #   > [None,None] for states with no analytical solutions
    anni_params = []   # Array of parameters [N, L, K]
    
    for k in K_states:
        for l in L_states:
            anni_params.append([N,l,k]) 
            Hs.append(get_H(int(N), float(l), float(k), ring))
            
            # Append the known labels (phases of the model)
            if k == 0:
                if l < 1:
                    labels.append([0,1]) # Ferromagnetic
                else:
                    labels.append([1,1]) # Paramagnetic
            elif l == 0:
                if k < -.5:
                    labels.append([1,0]) # Ferromagnetic
                else:
                    labels.append([0,1]) # Antiphase
            else:
                labels.append([None,None])
    
    # Array of indices for the order of states to train through VQE
    #     INDICES                RECYCLE RULE
    # +--------------+       +--------------+
    # | 4  9  14  19 |       | 4  5  14  15 |
    # | 3  8  13  18 |       | 3  6  13  16 |
    # | 2  7  12  17 |  ==>  | 2  7  12  17 |
    # | 1  6  11  16 |       | 1  8  11  18 |
    # | 0  5  10  15 |       | 0  9  10  19 |
    # +--------------+       +--------------+
    recycle_rule = []
    k = 0
    while k < n_states:
        recycle_rule.append(np.arange(k*n_states, (k+1)*n_states) )
        k += 1
        if k >= n_states:
            break
        recycle_rule.append(np.arange((k+1)*n_states - 1, k*n_states - 1, -1) )
        k += 1
        
    return Hs, np.array(labels), np.array(recycle_rule).flatten(), np.array(anni_params)


def build_Hs2(N, n_states, ring = False):
    """
    Sets up np.ndarray of pennylane Hamiltonians with different parameters
    total_states = n_states * n_states
    Taking n_states values of K from 0 to -1 (NB the sign)
    Taking n_states values of L from 0 to +2
    
    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    n_states : int
        Number of Hamiltonians to generate
    ring : bool
        If False, system has open-boundaries condition
        

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
    
    # Set up arrays of the parameters K and Ls
    K_states = np.linspace(0, -1, n_states)
    L_states = np.linspace(0, 2, n_states)
    
    Hs = []            # Array of the Pennylane hamiltonians
    labels = []        # Array of the labels:
                       #   > [1,1] for paramagnetic states
                       #   > [0,1] for ferromagnetic states
                       #   > [1,0] for antiphase states
                       #   > [None,None] for states with no analytical solutions
    anni_params = []   # Array of parameters [N, L, K]
    
    for k in K_states:
        for l in L_states:
            anni_params.append([N,l,k]) 
            Hs.append(get_H(int(N), float(l), float(k), ring))
            
            # Append the known labels (phases of the model)
            if k == 0:
                if l < 1:
                    labels.append([0,1]) # Ferromagnetic
                else:
                    labels.append([1,1]) # Paramagnetic
            elif l == 0:
                if k < -.5:
                    labels.append([1,0]) # Ferromagnetic
                else:
                    labels.append([0,1]) # Antiphase
            else:
                labels.append([None,None])
    
    # Array of indices for the order of states to train through VQE
    #     INDICES                RECYCLE RULE
    # +--------------+       +--------------+
    # | 4  9  14  19 |       | 4  5  14  15 |
    # | 3  8  13  18 |       | 3  6  13  16 |
    # | 2  7  12  17 |  ==>  | 2  7  12  17 |
    # | 1  6  11  16 |       | 1  8  11  18 |
    # | 0  5  10  15 |       | 0  9  10  19 |
    # +--------------+       +--------------+
    recycle_rule = []
    k = 0
    while k < n_states:
        recycle_rule.append(np.arange(k*n_states, (k+1)*n_states) )
        k += 1
        if k >= n_states:
            break
        recycle_rule.append(np.arange((k+1)*n_states - 1, k*n_states - 1, -1) )
        k += 1
        
    return Hs, np.array(labels), np.array(recycle_rule).flatten(), np.array(anni_params)