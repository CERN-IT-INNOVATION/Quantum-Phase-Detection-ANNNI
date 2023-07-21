"""Module for generic functions for other modules"""
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp

from typing import List, Tuple, Union
from numbers import Number


def linalgeigh(mat_H: List[List[Number]]) -> Tuple[int, List[Number]]:
    """
    Apply jax np.linalg.eigh on a matrix,
    to be jitted
    
    Parameters
    ----------
    mat_H : np.ndarray
        Input matrix to apply np.linalg.eigh

    Returns
    -------
    np.ndarray
        Array of eigenvalues (not sorted)
    np.ndarray
        Array of relative eigenvectors
    """
    # Compute eigenvalues and eigenvectors
    eigval, eigvec = jnp.linalg.eigh(mat_H)

    return eigval, eigvec


j_linalgeigh = jax.jit(linalgeigh)


def geteigvals(
    qml_H: qml.ops.qubit.hamiltonian.Hamiltonian, states: List[int]
) -> List[Number]:
    """
    Function for getting the energy values of an Ising Hamiltonian
    using the jitted jnp.linalg.eigh function
        
    Parameters
    ----------
    qml_H : pennylane.ops.qubit.hamiltonian.Hamiltonian
        Pennylane Hamiltonian of the state
    states : list
        List of energy levels desired

    Returns
    -------
    list
        List of energy values
    """

    # Get the matricial for of the hamiltonian
    # and convert it to float32
    # This type of hamiltonians are always real
    mat_H = np.real(qml.matrix(qml_H)).astype(np.single)

    # Compute sorted eigenvalues with jitted function
    eigvals = jnp.sort(j_linalgeigh(mat_H)[0])

    return [eigvals[k] for k in states]


def get_H_eigval_eigvec(
    qml_H: qml.ops.qubit.hamiltonian.Hamiltonian, en_lvl: int
) -> Tuple[List[List[Number]], Number, List[Number]]:
    """
    Function for getting the energy value and state of an Ising Hamiltonian
    using the jitted jnp.linalg.eigh function
        
    Parameters
    ----------
    qml_H : pennylane.ops.qubit.hamiltonian.Hamiltonian
        Pennylane Hamiltonian of the state
    en_lvl : int
        Energy level desired

    Returns
    -------
    np.ndarray
        Matricial encoding of the Hamiltonian
    float
        Value of the energy level
    np.ndarray
        Eigenstate of the energy level
    """

    # Get the matricial for of the hamiltonian
    # and convert it to float32
    # This type of hamiltonians are always real
    mat_H = qml.matrix(qml_H)

    # Compute sorted eigenvalues with jitted function
    eigvals, eigvecs = j_linalgeigh(mat_H)

    psi = eigvecs[:, jnp.argsort(eigvals)[en_lvl]]
    en = jnp.sort(eigvals)[en_lvl]

    return mat_H, en, psi


def psi_outer(psi: List[Number]) -> List[List[Number]]:
    return jnp.outer(jnp.conj(psi), psi)


j_psi_outer = jax.jit(psi_outer)
jv_psi_outer = jax.jit(jax.vmap(psi_outer))


def get_VQE_params(
    qml_H: qml.ops.qubit.hamiltonian.Hamiltonian,
) -> Tuple[List[List[Number]], Number]:
    """
    Function for getting all the training parameter for the VQE
    algorithm
        
    Parameters
    ----------
    qml_H : pennylane.ops.qubit.hamiltonian.Hamiltonian
        Pennylane Hamiltonian of the state
        
    Returns
    -------
    np.ndarray
        Matricial encoding of the Hamiltonian
    float
        Ground-state energy value
    """

    # Get the matricial for of the hamiltonian
    # and convert it to float32
    # This type of hamiltonians are always real
    mat_H = np.real(qml.matrix(qml_H)).astype(np.single)

    # Compute sorted eigenvalues with jitted function
    eigvals = j_linalgeigh(mat_H)[0]

    en0 = jnp.sort(eigvals)[0]

    return jnp.array([mat_H]), en0


def get_VQD_params(
    qml_H: qml.ops.qubit.hamiltonian.Hamiltonian, beta: Number
) -> Tuple[List[List[Number]], List[List[Number]], Number]:
    """
    Function for getting all the training parameter for the VQD
    algorithm for finding the first excited state
        
    Parameters
    ----------
    qml_H : pennylane.ops.qubit.hamiltonian.Hamiltonian
        Pennylane Hamiltonian of the Ising Model
        
    Returns
    -------
    np.ndarray
        Matricial encoding of the Hamiltonian
    np.ndarray
        Effective Hamiltonian of VQD algorithm
    float
        Excited-state energy value
    """

    # Get the matricial for of the hamiltonian
    # and convert it to float32
    # This type of hamiltonians are always real
    mat_H = np.real(qml.matrix(qml_H)).astype(np.single)

    # Compute sorted eigenvalues with jitted function
    eigvals, eigvecs = j_linalgeigh(mat_H)

    psi0 = eigvecs[:, jnp.argsort(eigvals)[0]]
    en_ex = jnp.sort(eigvals)[1]

    return jnp.array([mat_H]), jnp.array([mat_H + beta * j_psi_outer(psi0)]), en_ex


def paraanti(x):
    return 1.05 * np.sqrt((x - 0.5) * (x - 0.1))


def paraferro(x):
    return ((1 - x) / x) * (1 - np.sqrt((1 - 3 * x + 4 * x * x) / (1 - x)))


def b1(x):
    return 1.05 * (x - 0.5)


def peshel_emery(x):
    y = (1 / (4 * x)) - x

    y[y > 2] = 2
    return y


def simple_to_idx(simple: int, side: int) -> Union[int, None]:
    if simple <= 2 * side - 1:
        if simple <= side:
            return simple
        else:
            return side * (simple % side + 1)
    return None
