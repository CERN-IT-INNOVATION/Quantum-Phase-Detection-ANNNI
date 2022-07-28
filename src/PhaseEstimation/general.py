import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit


def linalgeigh(mat_H):
    # Compute eigenvalues and eigenvectors
    eigval, eigvec = jnp.linalg.eigh(mat_H)
    
    return eigval, eigvec

j_linalgeigh = jax.jit(linalgeigh)

def geteigvals(qml_H, states):
    # Get the matricial for of the hamiltonian
    # and convert it to float32 
    # This type of hamiltonians are always real
    mat_H = np.real(qml.matrix(qml_H)).astype(np.single)
    
    # Compute sorted eigenvalues with jitted function
    eigvals = jnp.sort(j_linalgeigh(mat_H)[0])
    
    return [eigvals[k] for k in states]

def get_H_eigval_eigvec(qml_H, en_lvl):
    # Get the matricial for of the hamiltonian
    # and convert it to float32 
    # This type of hamiltonians are always real
    mat_H = np.real(qml.matrix(qml_H)).astype(np.single)
    
    # Compute sorted eigenvalues with jitted function
    eigvals, eigvecs = j_linalgeigh(mat_H)
    
    psi = eigvecs[:,jnp.argsort(eigvals)[en_lvl]]
    en = jnp.sort(eigvals)[en_lvl]
    
    return mat_H, en, psi

def psi_outer(psi):
    return jnp.outer(jnp.conj(psi), psi)

j_psi_outer = jax.jit(psi_outer)
jv_psi_outer = jax.jit(jax.vmap(psi_outer))

def get_neighbours(vqeclass, idx):
    side = int(np.sqrt(vqeclass.n_states))
    
    neighbours = np.array([idx + 1, idx - 1, idx + side, idx - side])
    neighbours = np.delete(neighbours, np.logical_not(np.isin(neighbours, vqeclass.Hs.recycle_rule)) )


    if (idx + 1) % side == 0 and idx != self.n_states - 1:
        neighbours = np.delete(neighbours, 0)
    if (idx    ) % side == 0 and idx != 0:
        neighbours = np.delete(neighbours, 1)
        
    return neighbours
