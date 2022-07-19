import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from matplotlib import pyplot as plt

import copy
import tqdm  # Pretty progress bars
import joblib  # Writing and loading

import sys
sys.path.insert(0, '../')
import vqe
import ising_chain
import annni_model
import hamiltonians
import visualization as qplt
import losses
##############

Hs = hamiltonians.hamiltonian(annni_model.build_Hs, N = 8, n_states = 100, ring = False)
myvqe = vqe.vqe(Hs, vqe.circuit_ising)
myvqe.train(.25, 100, circuit = True, recycle = True, reg = 0)
myvqe.train(.25, 10000, circuit = False, recycle = False, reg = 100)

myvqe.save('./vqe_annni.pkl')