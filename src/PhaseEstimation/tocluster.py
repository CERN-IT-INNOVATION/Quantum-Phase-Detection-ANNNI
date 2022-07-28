import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

import sys
sys.path.insert(0, '../')
import vqe as vqe
import annni_model
import hamiltonians
##############

print('Initializing Hamiltonians')
Hs = hamiltonians.hamiltonian(annni_model.build_Hs, N = 8, n_states = 10, ring = False)

print('Initializing VQE class')
myvqe = vqe.vqe(Hs, vqe.circuit_ising)

print('Training (recycle):')
myvqe.train(.3, 50, circuit = True, recycle = True, reg = 0)

print('Training (parallel):')
myvqe.train(.3, 5000, circuit = False, recycle = False, reg = 10, batch_size = 250)

print('Saving:')
myvqe.save('../vqe_annni.pkl')
