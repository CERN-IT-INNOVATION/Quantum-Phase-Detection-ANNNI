""" This module implements the base class for spin-models Hamiltonians"""
import pennylane as qml
from pennylane import numpy as np
import jax.numpy as jnp

##############

class hamiltonian:
    def __init__(self, building_func, **kwargs):
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
        self.qml_Hs, self.labels, self.recycle_rule, self.model_params = self.func(**kwargs)
        
        # Get the groundstate-energy and the matrix hamiltonian
        true_gs_en = []
        true_ex_en = []
        
        for qml_H in self.qml_Hs:
            # True groundstate energies
            eigs = np.sort(qml.eigvals(qml_H))
            true_gs_en.append(eigs[0])
            true_ex_en.append(eigs[1])
            
        self.true_e0 = jnp.array(true_gs_en)
        self.true_e1 = jnp.array(true_ex_en)
        self.n_states = len(self.qml_Hs)
            
