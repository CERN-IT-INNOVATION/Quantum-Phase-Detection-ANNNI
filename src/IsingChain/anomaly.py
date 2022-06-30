""" This module implements the base function to implement a VQE for a Ising Chain with Transverse Field. """
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
from noisyopt import minimizeSPSA

import multiprocessing

import warnings

warnings.filterwarnings(
    "ignore",
    message="For Hamiltonians, the eigenvalues will be computed numerically. This may be computationally intensive for a large number of wires.Consider using a sparse representation of the Hamiltonian with qml.SparseHamiltonian.",
)

from . import vqe as vqe

##############

def circuit_wall_RY(N, param, index=0):
    """
    Apply independent RY rotations to each wire in a Pennylane circuit

    Parameters
    ----------
    N : int
        Number of qubits
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    # Apply RY to each wire:
    for spin in range(N):
        qml.RY(param[index + spin], wires=spin)

    return index + N

def circuit_anomaly_entanglement(N, wires, wires_trash):
    """
    Applies CX between a wire and a trash wire for each 
    wire/trashwire
    
    Parameters
    ----------
    N : int
        Number of qubits
    wires : np.ndarray
        Array of the indexes of non-trash qubits
    wires_trash : np.ndarray
        Array of the indexes of trash qubits (np.1dsetdiff(np.arange(N),wires))
    """
    # Connection between trash wires
    for wire, wire_next in zip(wires_trash[0::2], wires_trash[1::2]):
        qml.CNOT(wires = [int(wire), int(wire_next)])
    for wire, wire_next in zip(wires_trash[1::2], wires_trash[2::2]):
        qml.CNOT(wires = [int(wire), int(wire_next)])
        
    # Connections wires -> trash_wires
    for trash_idx, wire in enumerate(wires):
        trash_idx = 0 if trash_idx > len(wires_trash) else trash_idx
        qml.CNOT(wires = [int(wire), wires_trash[trash_idx]])


def anomaly_circuit(N, vqe_circuit, vqe_params, params):
    """
    Building function for the circuit:
          VQE(params_vqe) + Anomaly(params)
    
    Parameters
    ----------
    N : int
        Number of qubits
    vqe_circuit : function
        Function of the VQE Circuit
    vqe_params : np.ndarray
        Array of VQE parameters (states)
    params: np.ndarray
        Array of parameters/rotation for the circuit
        
    Returns
    -------
    np.ndarray
        Index of the trash qubits
    int
        Number of parameters of the circuit    
    """
    # Number of wires that will not be measured |phi>
    n_wires = N//2 + N%2
    # Number of wires that will be measured |0>^k
    n_trash = N//2
    
    wires = np.concatenate((np.arange(0, n_wires//2 + n_wires%2), np.arange(N-n_wires//2,N) ))
    wires_trash = np.setdiff1d(np.arange(N), wires)

    vqe_circuit(N, vqe_params)
    
    # Visual Separation VQE||Anomaly
    qml.Barrier()
    qml.Barrier()
    index = circuit_wall_RY(N, params)
    circuit_anomaly_entanglement(N, wires, wires_trash)
    qml.Barrier()
    index = circuit_wall_RY(N, params, index)
    circuit_anomaly_entanglement(N, wires, wires_trash)
    qml.Barrier()
    index = circuit_wall_RY(N, params, index)
    
    return wires_trash, index

def train(epochs, lr, N, vqe_circuit, X_train, X_test, train_index, plot = False, circuit = False):
    #X_train, Y_train = jnp.array(X_train), jnp.array(Y_train)

    device = qml.device("default.qubit.jax", wires = N, shots = None)
    @qml.qnode(device, interface="jax")
    def encoder_circuit(N, vqe_circuit, vqe_params, params):
        wires_trash, _ = anomaly_circuit(N, vqe_circuit, vqe_params, params)

        # return <psi|H|psi>
        return [qml.expval(qml.PauliZ(k)) for k in wires_trash]
    
    v_encoder_circuit = jax.vmap(lambda p, x: encoder_circuit(N, vqe_circuit, x, p), in_axes = (None, 0) )
    
    def compress(params, vqe_params):
        return - jnp.sum(- 1 - v_encoder_circuit(params, vqe_params) )/(2*len(vqe_params))
    
    wires_trash, n_params = anomaly_circuit(N, vqe_circuit, X_train[0], [0]*100)
    
    if circuit:
        print('+-- CIRCUIT ---+')
        drawer = qml.draw(encoder_circuit)
        print(np.arange(n_params))
        print(drawer(N, vqe_circuit, [0]*len(X_train[0]), np.arange(n_params)) )
    
    jd_compress     = jax.jit(jax.grad(lambda p: compress(p, X_train)) )
    j_compress      = jax.jit(lambda p: compress(p, X_train))
    get_compression = jax.jit(lambda p: jnp.sum(v_encoder_circuit(p, X_train),axis=1)/len(wires_trash) )
    
    if len(X_test) > 0:
        j_compress_test = jax.jit(lambda p: compress(p, X_train))
        get_compression_test = jax.jit(lambda p: jnp.sum(v_encoder_circuit(p, X_test),axis=1)/len(wires_trash) )
    
    params = np.random.rand(n_params)
    
    loss = []
    progress = tqdm.tqdm(range(epochs), position=0, leave=True)
    for epoch in range(epochs):
        params -= lr*jd_compress(params)
        
        if (epoch+1) % 100 == 0:
            loss.append(j_compress(params))
            progress.set_description('Cost: {0}'.format(loss[-1]) )
        progress.update(1)

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(16,5))
        
        ax[0].scatter(train_index, get_compression(params), label = 'Training')
        if len(X_test) > 0:
            ax[0].scatter(np.setdiff1d(np.arange(len(X_train)+len(X_test)), train_index), get_compression_test(params) , label = 'Test')
        ax[0].axvline(x=len(data)//2, color='red', linestyle='--')    
        ax[0].legend()
        ax[0].grid(True)
        
        ax[1].set_title('Loss of the encoder')
        ax[1].plot(np.arange(len(loss))*100, loss)
        
        
    return params
