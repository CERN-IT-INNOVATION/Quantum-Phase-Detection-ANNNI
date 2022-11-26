""" This module implements the base class for spin-models Hamiltonians"""

from PhaseEstimation import general as qmlgen

from tqdm.auto import tqdm
from typing import Callable
import numpy as np
##############


class hamiltonian:
    N: int
    J: float

    def __init__(self, building_func: Callable, **kwargs):
        """
        Hamiltonian class

        Parameters
        ----------
        building_func : function
            Function for preparing the hamiltonians of the model
        **kwags : arguments
            Arguments of the building_func function
        """
        self.func = building_func

        # Set the kwargs to attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Get pennylane hamiltonians and labels
        (
            self.qml_Hs,
            self.labels,
            self.recycle_rule,
            self.model_params,
            self.n_states,
            self.n_hs,  self.n_kappas,
            self.h_max, self.kappa_max
        ) = self.func(**kwargs)

        self.n_states = len(self.qml_Hs)

def get_e_psi(Hclass, en_lvl):
    e_list   = []
    psi_list = []
    for H in tqdm(Hclass.qml_Hs):
        _, e, psi = qmlgen.get_H_eigval_eigvec(H, en_lvl)
        e_list.append(e), psi_list.append(psi)

    return np.array(e_list), np.array(psi_list)
