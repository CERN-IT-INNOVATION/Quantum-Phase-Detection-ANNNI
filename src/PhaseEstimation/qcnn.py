""" This module implements the base functions to implement a VQE for a Ising Chain with Transverse Field. """
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from matplotlib import pyplot as plt

import copy
import tqdm  # Pretty progress bars
import joblib  # Writing and loading

import warnings

warnings.filterwarnings(
    "ignore",
    message="For Hamiltonians, the eigenvalues will be computed numerically. This may be computationally intensive for a large number of wires.Consider using a sparse representation of the Hamiltonian with qml.SparseHamiltonian.",
)

import sys, os
sys.path.insert(0, '../../')
import PhaseEstimation.vqe

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


class qcnn:
    def __init__(self, vqe, qcnn_circuit):
        """
        Class for the QCNN algorithm

        Parameters
        ----------
        vqe : class
            VQE class
        qcnn_circuit :
            Function of the QCNN circuit
        """
        self.N = vqe.N
        self.J = vqe.J
        self.n_states = vqe.n_states
        self.circuit = lambda vqe_p, qcnn_p: qcnn_circuit(
            vqe_p, vqe.circuit_fun, qcnn_p, self.N
        )
        self.n_params = self.circuit([0] * 10000, [0] * 10000)
        self.params = np.array([np.pi / 4] * self.n_params)
        self.device = vqe.device

        self.vqe_states = np.array(vqe.vqe_states)
        self.lams = np.linspace(0, 2 * self.J, self.n_states)
        self.labels = np.array(vqe.labels)

        self.loss_train = []
        self.loss_test = []

        self.circuit_fun = qcnn_circuit

    def show_circuit(self):
        """
        Prints the current circuit defined by self.circuit
        """

        @qml.qnode(self.device, interface="jax")
        def qcnn_state(self):
            self.circuit([0] * 1000, np.arange(self.n_params))

            return qml.state()

        drawer = qml.draw(qcnn_state)
        print(drawer(self))

    # Training function
    def train(self, lr, n_epochs, train_index, circuit=False, plot=False):
        """
        Training function for the QCNN.

        Parameters
        ----------
        lr : float
            Learning rate to be multiplied in the circuit-gradient output
        n_epochs : int
            Total number of epochs for each learning
        train_index : np.ndarray
            Index of training points
        circuit : bool
            if True -> Prints the circuit
        plot : bool
            if True -> It displays loss curve
        """

        X_train, Y_train = jnp.array(self.vqe_states[train_index]), jnp.array(
            self.labels[train_index]
        )
        test_index = np.setdiff1d(np.arange(len(self.vqe_states)), train_index)
        X_test, Y_test = jnp.array(self.vqe_states[test_index]), jnp.array(
            self.labels[test_index]
        )

        if circuit:
            # Display the circuit
            print("+--- CIRCUIT ---+")
            self.show_circuit()

        @qml.qnode(self.device, interface="jax")
        def qcnn_circuit_prob(params_vqe, params):
            self.circuit(params_vqe, params)

            return qml.probs(wires=self.N - 1)

        def compute_cross_entropy(X, Y, params):
            v_qcnn_prob = jax.vmap(lambda v: qcnn_circuit_prob(v, params))

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

        params = copy.copy(self.params)

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

        self.loss_train = loss_history
        self.loss_test = loss_history_test
        self.params = params

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

    def show_results(self, train_index):
        """
        Plots performance of the classifier on the whole data

        Parameters
        ----------
        train_index : np.ndarray
            Index of training points
        """

        @qml.qnode(self.device, interface="jax")
        def qcnn_circuit_prob(params_vqe, params):
            self.circuit(params_vqe, params)

            return qml.probs(wires=self.N - 1)

        test_index = np.setdiff1d(np.arange(len(self.vqe_states)), train_index)

        predictions_train = []
        predictions_test = []

        colors_train = []
        colors_test = []

        vcircuit = jax.vmap(lambda v: qcnn_circuit_prob(v, self.params), in_axes=(0))
        predictions = vcircuit(self.vqe_states)[:, 1]

        for i, prediction in enumerate(predictions):
            # if data in training set
            if i in train_index:
                predictions_train.append(prediction)
                if np.round(prediction) == 0:
                    colors_train.append("green") if self.labels[
                        i
                    ] == 0 else colors_train.append("red")
                else:
                    colors_train.append("red") if self.labels[
                        i
                    ] == 0 else colors_train.append("green")
            else:
                predictions_test.append(prediction)
                if np.round(prediction) == 0:
                    colors_test.append("green") if self.labels[
                        i
                    ] == 0 else colors_test.append("red")
                else:
                    colors_test.append("red") if self.labels[
                        i
                    ] == 0 else colors_test.append("green")

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
            2 * np.sort(train_index) / len(self.vqe_states),
            predictions_train,
            c="royalblue",
            label="Training samples",
        )
        ax[0].scatter(
            2 * np.sort(test_index) / len(self.vqe_states),
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
        ax[1].scatter(
            2 * np.sort(train_index) / len(self.vqe_states),
            predictions_train,
            c=colors_train,
        )
        ax[1].scatter(
            2 * np.sort(test_index) / len(self.vqe_states),
            predictions_test,
            c=colors_test,
        )

    def save(filename):
        """
        Saves QCNN parameters to file

        Parameters
        ----------
        filename : str
            File where to save the parameters
        """

        things_to_save = [self.params, self.circuit_fun]

        with open(filename, "wb") as f:
            pickle.dump(things_to_save, f)


def load(filename_vqe, filename_qcnn):
    """
    Load QCNN from VQE file and QCNN file
    
    Parameters
    ----------
    filename_vqe : str
        Name of the file from where to load the VQE class
    filename_qcnn : str
        Name of the file from where to load the main parameters of the QCNN class
    """
    loaded_vqe = vqe.load(filename_vqe)

    with open(filename_qcnn, "rb") as f:
        params, qcnn_circuit_fun = pickle.load(f)

    loaded_qcnn = qcnn(vqe, qcnn_circuit_fun)
    loaded_qcnn.params = params

    return loaded_qcnn
