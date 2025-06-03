"""
Mathematical utility functions.
"""

import numpy as np

def fidelity(state1 : np.ndarray, state2 : np.ndarray) -> float:
    """
    Compute the fidelity of two quantum states, given as 1D arrays or vector.

    The fidelity is defined as the absolute square of the inner product of the two
    states, i.e. F(state1, state2) = |<state1|state2>|^2.

    Parameters
    ----------
    state1, state2 : 1D arrays or vectors
        The two quantum states to compute the fidelity of.

    Returns
    -------
    fidelity : float
        The fidelity of the two states.
    """
    return np.abs(np.vdot(state1, state2)) ** 2

def gram_schmidt(matrix: np.ndarray) -> np.ndarray:
    unitary = np.zeros(matrix.shape, dtype=matrix.dtype)

    p_orthonormal_vec = []
    for i in range(matrix.shape[1]):
        col = matrix[:, i].copy()
        if not np.allclose(col, np.zeros(len(col))):
            p_orthonormal_vec += [col]

    for j in range(matrix.shape[1]):
        basis = np.array(p_orthonormal_vec)
        col = matrix[:, j].copy()
        if np.allclose(col, np.zeros(len(col))):
            real = np.random.uniform(-1, 1, len(col))
            imag = np.random.uniform(-1, 1, len(col)) if np.iscomplexobj(unitary) else 0
            col = real + 1j * imag

            for vec in basis:
                col -= (vec.conj().T @ col) * vec / np.linalg.norm(vec)

        unitary[:, j] = col / np.linalg.norm(col)
        p_orthonormal_vec += [unitary[:, j]]

    return unitary