""" This module implements the base class for spin-models Hamiltonians"""

import general as qmlgen, annni_model as annni
import warnings 
import tqdm
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

    def add_true(self):
        """
        Add true ground-state energy levels and true wavefunctions by diagonalizing the Hamiltonian matrices
        """
        
        # Checks wether this has already been computed
        try:
            _,_, = self.true_e0, self.true_psi0
        except:
            warnings.warn("True Wavefunction and Groundstate energy levels not found, they will be not computed (this may take a while...)")
            self.true_e0, self.true_psi0 = get_e_psi(self, 0)

        # Checks wether this has already been computed
        try:
            _,_, = self.true_e1, self.true_psi1
        except:
            warnings.warn("True Wavefunction and First excited energy levels not found, they will be not computed (this may take a while...)")
            self.true_e1, self.true_psi1 = get_e_psi(self, 1)

def get_e_psi(Hclass, en_lvl):
    """
    Return respectively the list of the true energies and true states obtained through the diagonalization of the hamiltonian matrices

    Parameters
    ----------
    Hclass : hamiltonians.hamiltonians
        Custom hamiltonian class
    en_lvl : int
        Energy level to inspect

    Returns
    -------
    List[Number]
        Array of the energies
    List[List[Number]]
        Array of the state vectors
    """
    e_list   = []
    psi_list = []

    progress = tqdm.tqdm(range(len(Hclass.qml_Hs)))
    for H in Hclass.qml_Hs:
        _, e, psi = qmlgen.get_H_eigval_eigvec(H, en_lvl)
        e_list.append(e), psi_list.append(psi)
        
        progress.update(1)

    return np.array(e_list), np.array(psi_list)
