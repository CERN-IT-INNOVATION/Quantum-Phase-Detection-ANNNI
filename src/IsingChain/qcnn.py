""" This module implements the base functions to implement a VQE for a Ising Chain with Transverse Field. """
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


def circuit_convolution(active_wires, params, N, index):
    """
    Convolution block for the QCNN

    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured during a previous pooling
    params: np.ndarray
        Array of parameters/rotation for the circuit
    N : int
        Number of qubits
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    if len(active_wires) > 1:
        # Convolution:
        for wire in active_wires:
            qml.RX(params[index], wires=int(wire))
            qml.RY(params[index + 1], wires=int(wire))
            index = index + 2

        # ---- > Establish entanglement: odd connections
        for wire, wire_next in zip(active_wires[0::2], active_wires[1::2]):
            qml.CNOT(wires=[int(wire), int(wire_next)])
            qml.RX(params[index], wires=int(wire))
            index = index + 1

        # ---- > Establish entanglement: even connections
        for wire, wire_next in zip(active_wires[1::2], active_wires[2::2]):
            qml.CNOT(wires=[int(wire), int(wire_next)])
            qml.RX(params[index], wires=int(wire))
            index = index + 1

    qml.RX(params[index], wires=N - 1)

    return index + 1


def circuit_pooling(active_wires, params, N, index):
    """
    Pooling block for the QCNN
    
    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured during a previous pooling
    params: np.ndarray
        Array of parameters/rotation for the circuit
    N : int
        Number of qubits
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    np.ndarray
        Updated array of active wires (not measured)
    """
    # Pooling:
    isodd = True if len(active_wires) % 2 != 0 else False

    for wire_meas, wire_next in zip(active_wires[0::2], active_wires[1::2]):
        m_0 = qml.measure(int(wire_meas))
        qml.cond(m_0 == 0, qml.RY)(params[index], wires=int(wire_next))
        qml.cond(m_0 == 1, qml.RY)(params[index + 1], wires=int(wire_next))
        index = index + 2

        # Removing measured wires from active_wires:
        active_wires = np.delete(active_wires, np.where(active_wires == wire_meas))

    # ---- > If the number of wires is odd, the last wires is not pooled
    #        so we apply a Y gate
    if isodd:
        qml.RY(params[index], wires=N - 1)
        index = index + 1

    return index, active_wires


def qcnn_circuit(params_vqe, vqe_circuit_fun, params, N):
    """
    Building function for the circuit:
          VQE(params_vqe) + QCNN(params)

    Parameters
    ----------
    params_vqe : np.ndarray
        Array of VQE parameters (states)
    vqe_circuit_fun : function
        Function of the VQE circuit
    params : np.ndarray
        Array of QCNN parameters
    N : int
        Number of qubits

    Returns
    -------
    int
        Total number of parameters needed to build this circuit
    """

    # Wires that are not measured (through pooling)
    active_wires = np.arange(N)

    # Input: State through VQE
    vqe_circuit_fun(N, params_vqe)

    # Visual Separation VQE||QCNN
    qml.Barrier()
    qml.Barrier()

    # Index of the parameter vector
    index = 0

    # Iterate Convolution+Pooling until we only have a single wires
    while len(active_wires) > 1:
        index = circuit_convolution(active_wires, params, N, index)
        qml.Barrier()
        index, active_wires = circuit_pooling(active_wires, params, N, index)
        qml.Barrier()
        index = circuit_convolution(active_wires, params, N, index)
        qml.Barrier()
        index, active_wires = circuit_pooling(active_wires, params, N, index)
        qml.Barrier()

    # Return the number of parameters
    return index + 1


# Training function
def train(
    lr,
    n_epochs,
    N,
    vqe_circuit_fun,
    qcnn_circuit_fun,
    X_train,
    Y_train,
    X_test=[],
    Y_test=[],
    circuit=False,
    plot=False,
):
    """
    Training function for the QCNN.

    Parameters
    ----------
    lr : float
        Learning rate to be multiplied in the circuit-gradient output
    n_epochs : int
        Total number of epochs for each learning
    N : int
        Number of spins/qubits
    vqe_circuit_fun : function
        Function of the VQE circuit
    qcnn_circuit_fun : function
        Function of the VQE + QCNN
    X_train : np.ndarray
        Array of the VQE parameters (training)
    Y_train : np.ndarray
        Array of the VQE labels (training)
    X_test : np.ndarray
        Array of the VQE parameters (tests)
    Y_test : np.ndarray
        Array of the VQE labels (tests)
    circuit : bool
        if True -> Prints the circuit
        if False -> It does not display the circuit
    plots : bool
        if True -> Display plots
        if False -> It does not display plots

    Returns
    -------
    np.ndarray
        Array of QCNN params
    """

    X_train, Y_train = jnp.array(X_train), jnp.array(Y_train)
    X_test, Y_test = jnp.array(X_test), jnp.array(Y_test)

    device = qml.device("default.qubit.jax", wires=N, shots=None)

    @qml.qnode(device, interface="jax")
    def qcnn_circuit_prob(params_vqe, params, N):
        qcnn_circuit_fun(params_vqe, vqe_circuit_fun, params, N)

        return qml.probs(wires=N - 1)

    def compute_cross_entropy(X, Y, params):
        v_qcnn_prob = jax.vmap(lambda v: qcnn_circuit_prob(v, params, N))

        predictions = v_qcnn_prob(X)
        logprobs = jnp.log(predictions)

        nll = jnp.take_along_axis(logprobs, jnp.expand_dims(Y, axis=1), axis=1)
        ce = -jnp.mean(nll)

        return ce

    # Gradient of the Loss function
    d_compute_cross_entropy = jax.jit(
        jax.grad(lambda p: compute_cross_entropy(X_train, Y_train, p))
    )

    # Compute Loss of whole sets
    train_compute_cross_entropy = jax.jit(
        lambda p: compute_cross_entropy(X_train, Y_train, p)
    )
    test_compute_cross_entropy = jax.jit(
        lambda p: compute_cross_entropy(X_test, Y_test, p)
    )

    # Check the number of parameters needed for the QCNN circuit
    n_params = qcnn_circuit_fun([0] * 1000, vqe_circuit_fun, [0] * 1000, N)
    # Initialize parameters
    params = np.array([np.pi / 4] * n_params)

    progress = tqdm.tqdm(range(n_epochs), position=0, leave=True)

    loss_history = []
    loss_history_test = []
    for epoch in range(n_epochs):
        params -= lr * d_compute_cross_entropy(params)

        if epoch % 100 == 0:
            loss_history.append(train_compute_cross_entropy(params))
            if len(Y_test) > 0:
                loss_history_test.append(test_compute_cross_entropy(params))
        progress.update(1)
        progress.set_description("Cost: {0}".format(loss_history[-1]))

    if plot:
        plt.figure(figsize=(15, 5))
        plt.plot(
            np.arange(len(loss_history)) * 100,
            np.asarray(loss_history),
            label="Training Loss",
        )
        if len(X_test) > 0:
            plt.plot(
                np.arange(len(loss_history_test)) * 100,
                np.asarray(loss_history_test),
                label="Test Loss",
            )
        plt.axhline(y=0, color="r", linestyle="--")
        plt.title("Loss history")
        plt.ylabel("Average Cross entropy")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend()

    return params


def plot_results(X, Y, train_index, params, N, vqe_circuit_fun, qcnn_circuit_fun):
    """
    Plots performance of the classifier on the whole data

    Parameters
    ----------
    X : np.ndarray
       Array of VQE parameters
    Y : np.ndarray
       Array of VQE labels
    train_index : np.ndarray
       Array of the indexes of the training set
    params : np.ndarray
       Array of QCNN parameters
    vqe_circuit_fun : function
        Function of the VQE circuit
    qcnn_circuit_fun : function
        Function of the VQE + QCNN
    """

    device = qml.device("default.qubit.jax", wires=N, shots=None)

    @qml.qnode(device, interface="jax")
    def qcnn_circuit_prob(params_vqe, params, N):
        qcnn_circuit_fun(params_vqe, vqe_circuit_fun, params, N)

        return qml.probs(wires=N - 1)

    test_index = []
    for i in range(len(Y)):
        if not i in train_index:
            test_index.append(i)

    predictions_train = []
    predictions_test = []

    colors_train = []
    colors_test = []

    vcircuit = jax.vmap(lambda v: qcnn_circuit_prob(v, params, N), in_axes=(0))
    predictions = vcircuit(X)[:, 1]

    for i, prediction in enumerate(predictions):
        # if data in training set
        if i in train_index:
            predictions_train.append(prediction)
            if np.round(prediction) == 0:
                if i <= len(Y) / 2:
                    colors_train.append("green")
                else:
                    colors_train.append("red")
            else:
                if i <= len(Y) / 2:
                    colors_train.append("red")
                else:
                    colors_train.append("green")
        else:
            predictions_test.append(prediction)
            if np.round(prediction) == 0:
                if i <= len(Y) / 2:
                    colors_test.append("green")
                else:
                    colors_test.append("red")
            else:
                if i <= len(Y) / 2:
                    colors_test.append("red")
                else:
                    colors_test.append("green")

    fig, ax = plt.subplots(2, 1, figsize=(16, 10))

    ax[0].set_xlim(-0.1, 2.1)
    ax[0].set_ylim(0, 1)
    ax[0].grid(True)
    ax[0].axhline(y=0.5, color="gray", linestyle="--")
    ax[0].axvline(x=1, color="gray", linestyle="--")
    ax[0].text(0.375, 0.68, "I", fontsize=24, fontfamily="serif")
    ax[0].text(1.6, 0.68, "II", fontsize=24, fontfamily="serif")
    ax[0].set_xlabel("Transverse field")
    ax[0].set_ylabel("Prediction of label II")
    ax[0].set_title("Predictions of labels; J = 1")
    ax[0].scatter(
        2 * np.sort(train_index) / len(X),
        predictions_train,
        c="royalblue",
        label="Training samples",
    )
    ax[0].scatter(
        2 * np.sort(test_index) / len(X),
        predictions_test,
        c="orange",
        label="Test samples",
    )
    ax[0].legend()

    ax[1].set_xlim(-0.1, 2.1)
    ax[1].set_ylim(0, 1)
    ax[1].grid(True)
    ax[1].axhline(y=0.5, color="gray", linestyle="--")
    ax[1].axvline(x=1, color="gray", linestyle="--")
    ax[1].text(0.375, 0.68, "I", fontsize=24, fontfamily="serif")
    ax[1].text(1.6, 0.68, "II", fontsize=24, fontfamily="serif")
    ax[1].set_xlabel("Transverse field")
    ax[1].set_ylabel("Prediction of label II")
    ax[1].set_title("Predictions of labels; J = 1")
    ax[1].scatter(2 * np.sort(train_index) / len(X), predictions_train, c=colors_train)
    ax[1].scatter(2 * np.sort(test_index) / len(X), predictions_test, c=colors_test)
