""" This module implements the base class for spin-models Hamiltonians"""
import pennylane as qml
from pennylane import numpy as np
import jax.numpy as jnp

import sys, os
sys.path.insert(0, '../../')
import PhaseEstimation.general as qmlgen

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
        
        self.n_states = len(self.qml_Hs)
            
