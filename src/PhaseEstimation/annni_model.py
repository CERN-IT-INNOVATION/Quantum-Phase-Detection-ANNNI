""" This module implements the base functions for treating the Hamiltonian of the ANNNI Ising-model """
import pennylane as qml
from pennylane import numpy as np

from typing import Tuple, List

##############


def get_H(
    N: int, L: float, K: float, ring: bool = False
) -> qml.ops.qubit.hamiltonian.Hamiltonian:
    """
    Set up Hamiltonian:
            H = J1* (- Σsigma^i_x*sigma_x^{i+1} - (h/J1) * Σsigma^i_z - (J2/J1) * Σsigma^i_x*sigma_x^{i+2} )
        
        [where J1 = 1, (h/J1) = Lambda(/L), (J2/J1) = K]

    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    L : float
        h/J1 parameter
    K : float
        J1/J2 parameter
    ring : bool
        If False, system has open-boundaries condition

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
        H = H + (-1) * K * (qml.PauliX(i) @ qml.PauliX(i + 2))

    # If ring == True, the 'chain' needs to be closed
    if ring:
        # Nearest interaction between last spin and first spin -particles
        H = H + (-1) * (qml.PauliX(N - 1) @ qml.PauliX(0))
        # Next nearest interactions:
        H = H + (-1) * K * (qml.PauliX(N - 2) @ qml.PauliX(0))
        H = H + (-1) * K * (qml.PauliX(N - 1) @ qml.PauliX(1))

    return H


def build_Hs(
    N: int,
    n_hs: int,
    n_kappas: int,
    h_max: float = 2,
    kappa_max: float = 1,
    ring: bool = False,
) -> Tuple[
    List[qml.ops.qubit.hamiltonian.Hamiltonian],
    List[List[int]],
    List[int],
    List[Tuple[int, float, float]],
    int,
]:
    """
    Sets up np.ndarray of pennylane Hamiltonians with different parameters
    total_states = n_kappas * n_hs
    kappa can have n_kappas values from 0 to - |kappa_max| (NB the sign)
    h     can have n_hs values from 0 to h_max
    
    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    n_hs : int
        Number of different values of h the hamiltonian can have
    h_max : float
        Maximum value of h, the values will range from 0 to h_max
    n_kappas : int
        Number of different values of kappa the hamiltonian can have
    kappa_max : float
        Maximum value of kappa, the values will range from 0 to - |kappa_max|
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
    kappa_values = np.linspace(0, -np.abs(kappa_max), n_kappas)
    h_values = np.linspace(0, h_max, n_hs)

    Hs = []  # Array of the Pennylane hamiltonians
    labels = []  # Array of the labels:
    #   > [1,1] for paramagnetic states
    #   > [0,1] for ferromagnetic states
    #   > [1,0] for antiphase states
    #   > [None,None] for states with no analytical solutions
    anni_params = []  # Array of parameters [N, L, K]

    for kappa in kappa_values:
        for h in h_values:
            anni_params.append([N, h, kappa])
            Hs.append(get_H(int(N), float(h), float(kappa), ring))

            # Append the known labels (phases of the model)
            if kappa == 0:
                if h < 1:
                    labels.append([0, 1])  # Ferromagnetic
                else:
                    labels.append([1, 1])  # Paramagnetic
            elif h == 0:
                if kappa < -0.5:
                    labels.append([1, 0])  # Antiphase
                else:
                    labels.append([0, 1])  # Ferromagnetic

            else:
                labels.append([-1, -1])

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
    k_index = 0
    while k_index < n_kappas:
        # k_index = 0 (going up)
        # [0, 1, 2, 3, 4]
        recycle_rule.append(np.arange(k_index * n_hs, (k_index + 1) * n_hs).astype(int))
        k_index += 1
        if k_index >= n_kappas:
            break
        # k_index = 1 (going down)
        # [9, 8, 7, 6, 5]
        recycle_rule.append(
            np.arange((k_index + 1) * n_hs - 1, k_index * n_hs - 1, -1).astype(int)
        )
        k_index += 1
        if k_index >= n_kappas:
            break

    return (
        Hs,
        np.array(labels),
        np.array(recycle_rule).flatten(),
        np.array(anni_params),
        n_hs * n_kappas,
        n_hs,
        n_kappas,
        h_max,
        kappa_max,
    )
