import numpy as np


def fidelity(state1, state2) -> float:
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
