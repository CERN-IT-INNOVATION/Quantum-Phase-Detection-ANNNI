""" This module implements the base function for treating ANNNI model """
import pennylane as qml
from pennylane import numpy as np
import jax

##############

def get_H(N, L, K):
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

    Returns
    -------
    pennylane.ops.qubit.hamiltonian.Hamiltonian
        Hamiltonian Pennylane class for the (Transverse) Ising Chain
    """
    # Interaction of spins with magnetic field
    H = -L * qml.PauliZ(0)
    for i in range(1, N):
        H = H - L * qml.PauliZ(i)

    # Interaction between spins (neighbouring):
    for i in range(0, N - 1):
        H = H + (-1) * (qml.PauliX(i) @ qml.PauliX(i + 1))
        
    # Interaction between spins (next-neighbouring):
    for i in range(0, N - 2):
        H = H + K * (-1) * (qml.PauliX(i) @ qml.PauliX(i + 2))

    return H

def build_Hs(N, n_states):
    """
    Sets up np.ndarray of pennylane Hamiltonians with different parameters
    total_states = n_states * n_states
    Taking n_states values of K from 0 to 1
    Taking n_states values of L from 0 to 2
    
    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    J : float
        Interaction strenght between spins
    n_states : int
        Number of Hamiltonians to generate
    """
    K_states = np.linspace(0, 1, n_states)
    L_states = np.linspace(0, 2, n_states)
    
    Hs = []
    labels = []
    anni_params = []
    for k in K_states:
        for l in L_states:
            anni_params.append([N,l,k])
            Hs.append(get_H(int(N), float(l), float(k)))
            
            # Append the known labels (phases of the model)
            if k == 0:
                if l < 1:
                    labels.append([0,1])
                else:
                    labels.append([1,0])
            elif l == 0:
                if k < .5:
                    labels.append([0,1])
                else:
                    labels.append([0,0])
            else:
                labels.append([None,None])
                
    recycle_rule = []
    k = 0
    while k < n_states:
        recycle_rule.append(np.arange(k*n_states, (k+1)*n_states) )
        k += 1
        if k >= n_states:
            break
        recycle_rule.append(np.arange((k+1)*n_states - 1, k*n_states - 1, -1) )
        k += 1
        
    return Hs, labels, np.array(recycle_rule).flatten(), anni_params
