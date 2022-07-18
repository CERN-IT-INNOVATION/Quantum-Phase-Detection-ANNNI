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
        mat_Hs = []
        true_e = []
        for qml_H in self.qml_Hs:
            # True groundstate energies
            true_e.append(np.min(qml.eigvals(qml_H)))
            # Standard matrix for of the hamiltonians
            mat_Hs.append(qml.matrix(qml_H))
        
        self.mat_Hs = jnp.array(mat_Hs)
        self.true_e = jnp.array(true_e)
        self.n_states = len(self.qml_Hs)
            
