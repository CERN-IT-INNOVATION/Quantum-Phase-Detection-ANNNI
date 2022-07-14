""" This module implements the base functions to implement a VQE for a Ising Chain with Transverse Field. """
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.example_libraries import optimizers

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
import PhaseEstimation.circuits as circuits

##############

def qcnn_circuit(params_vqe, vqe_circuit_fun, params, N, n_outputs):
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
    while len(active_wires) > n_outputs:
        index = circuits.convolution(active_wires, params, index)
        qml.Barrier()
        index, active_wires = circuits.pooling(active_wires, qml.RY, params, index)
        qml.Barrier()
        index = circuits.convolution(active_wires, params, index)
        qml.Barrier()
        index, active_wires = circuits.pooling(active_wires, qml.RY, params, index)
        qml.Barrier()

    # Return the number of parameters
    return index + 1


class qcnn:
    def __init__(self, vqe, qcnn_circuit, n_outputs = 1):
        """
        Class for the QCNN algorithm

        Parameters
        ----------
        vqe : class
            VQE class
        qcnn_circuit :
            Function of the QCNN circuit
        """
        self.vqe = vqe
        self.N = vqe.N
        self.n_states = vqe.n_states
        self.circuit = lambda vqe_p, qcnn_p: qcnn_circuit(
            vqe_p, vqe.circuit_fun, qcnn_p, self.N, n_outputs
        )
        self.n_params = self.circuit([0] * 10000, [0] * 10000)
        self.params = np.array([np.pi / 4] * self.n_params)
        self.device = vqe.device

        self.vqe_states = np.array(vqe.vqe_states)
        self.labels = np.array(vqe.Hs.labels)
        self.train_index = []
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
    def train(self, lr, n_epochs, train_index, loss_fn, circuit=False, plot=False):
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

        # Gradient of the Loss function
        jd_loss_fn = jax.jit(
            jax.grad(lambda p: loss_fn(X_train, Y_train, p, qcnn_circuit_prob))
        )

        def update(params, opt_state):
            grads = jd_loss_fn(params)
            opt_state = opt_update(0, grads, opt_state)
            
            return get_params(opt_state), opt_state
        
        # Compute Loss of whole sets
        train_loss_fn = jax.jit(
            lambda p: loss_fn(X_train, Y_train, p, qcnn_circuit_prob)
        )
        test_loss_fn = jax.jit(
            lambda p: loss_fn(X_test, Y_test, p, qcnn_circuit_prob)
        )

        params = copy.copy(self.params)

        progress = tqdm.tqdm(range(n_epochs), position=0, leave=True)

        # Defining an optimizer in Jax
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(params)
        
        loss_history = []
        loss_history_test = []
        for epoch in range(n_epochs):
            params, opt_state = update(params, opt_state)

            if epoch % 100 == 0:
                loss_history.append(train_loss_fn(params))
                if len(Y_test) > 0:
                    loss_history_test.append(test_loss_fn(params))
            progress.update(1)
            progress.set_description("Cost: {0}".format(loss_history[-1]))

        self.loss_train = loss_history
        self.loss_test = loss_history_test
        self.params = params
        self.train_index = train_index
        
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
