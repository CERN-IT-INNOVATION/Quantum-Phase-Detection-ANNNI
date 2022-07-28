""" This module implements the base function to implement a VQE for a Ising Chain with Transverse Field. """
import pennylane as qml
from pennylane import numpy as np

import vqe2 as vqe
import annni_model as annni
import hamiltonians as ham
import ising_chain as ising_chain
import visualization as qplt

N = 8
J = 1
l_steps = 100

Hs = ham.hamiltonian(annni.build_Hs, N = N, n_states = l_steps)
myvqe = vqe.vqe(Hs, vqe.circuit_ising)

myvqe.train(0.3, 300, circuit = False, recycle = False, epochs_batch_size = 100, batch_size = 100)
myvqe.train_refine(0.1, 10000, 0.01)

myvqe.save('./N'+str(N)+'n'+str(l_steps)+'.pkl')