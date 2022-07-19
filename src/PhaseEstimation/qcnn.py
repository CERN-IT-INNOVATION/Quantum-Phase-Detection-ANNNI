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

def qcnn_circuit(params, N, n_outputs):
    """
    Building function for the QCNN circuit:
    
    Parameters
    ----------
    params : np.ndarray
        Array of QCNN parameters
    N : int
        Number of qubits
    n_outputs : int
        Output vector dimension

    Returns
    -------
    int
        Total number of parameters needed to build this circuit
    """

    # Wires that are not measured (through pooling)
    active_wires = np.arange(N)

    # Visual Separation VQE||QCNN
    qml.Barrier()
    qml.Barrier()

    # Index of the parameter vector
    index = 0

    # Iterate Convolution+Pooling until we only have a single wires
    while len(active_wires) > n_outputs:
        index = circuits.convolution(active_wires, params, index)
        circuits.wall_gate(active_wires, qml.Hadamard)
        index = circuits.convolution(active_wires, params, index)
        qml.Barrier()
        index, active_wires = circuits.pooling(active_wires, qml.RZ, params, index)
        qml.Barrier()
    index = circuits.wall_gate(active_wires, qml.RX, params, index)
    index = circuits.wall_gate(active_wires, qml.RY, params, index)

    # Return the number of parameters
    return index + 1, active_wires

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
        n_outputs : int
            Output vector dimension
        """
        self.vqe = vqe
        self.N = vqe.N
        self.n_states = vqe.n_states
        self.n_outputs = n_outputs
        self.qcnn_circuit_fun = lambda p: qcnn_circuit(p, self.N, n_outputs)
        self.n_params, self.final_active_wires = self.qcnn_circuit_fun([0] * 10000)
        self.params = np.array(np.random.rand(self.n_params) )
        self.device = vqe.device

        self.vqe_params = np.array(vqe.vqe_params)
        self.labels = np.array(vqe.Hs.labels)
        self.train_index = []
        self.loss_train = []
        self.loss_test = []

        @qml.qnode(self.device, interface="jax")
        def circuit_drawer(self):
            _, active_wires = self.qcnn_circuit_fun(np.arange(self.n_params))
            if n_outputs == 1:
                return qml.probs(wires=self.N - 1)
            else:
                return [qml.probs(wires=int(k)) for k in active_wires]
            
        self.drawer = qml.draw(circuit_drawer)(self)
    
    def vqe_qcnn_circuit(self, vqe_p, qcnn_p):
        self.vqe.circuit(vqe_p)
        self.qcnn_circuit_fun(qcnn_p)
        
    def psi_qcnn_circuit(self, psi, qcnn_p):
        qml.QubitStateVector(psi, wires=[int(k) for k in range(self.N)])
        self.qcnn_circuit_fun(qcnn_p)
    
    # Training function
    def train(self, lr, n_epochs, train_index, loss_fn, circuit=False, plot=False, inject = False):
        """
        Training function for the QCNN.

        Parameters
        ----------
        lr : float
            Learning rate for the ADAM optimizer
        n_epochs : int
            Total number of epochs for each learning
        train_index : np.ndarray
            Index of training points
        loss_fn : function
            Loss function
        circuit : bool
            if True -> Prints the circuit
        plot : bool
            if True -> It displays loss curve
        inject : bool
            if True -> Exact ground states will be computed and used as input
        """

        if None not in self.labels:
            X_train, Y_train = jnp.array(self.vqe_params[train_index]), jnp.array(
                self.labels[train_index]
            )
            test_index = np.setdiff1d(np.arange(len(self.vqe_params)), train_index)
            X_test, Y_test = jnp.array(self.vqe_params[test_index]), jnp.array(
                self.labels[test_index]
            )
        else:
            mask = jnp.array(jnp.logical_or(jnp.array(self.vqe.Hs.model_params)[:,1] == 0, jnp.array(self.vqe.Hs.model_params)[:,2] == 0))
            #mask = jnp.array(self.vqe.Hs.model_params)[:,1] == 0
            self.vqe_params = jnp.array(self.vqe_params)
            
            #mask = jnp.array(myqcnn.vqe.Hs.model_params)[:,1] == 0
            X, Y = self.vqe_params[mask], self.labels[mask,:].astype(int)
            test_index = np.setdiff1d(np.arange(len(Y)), train_index)
            
            X_train, Y_train = X[train_index], Y[train_index]
            X_test, Y_test   = X[test_index], Y[test_index]
            
        if circuit:
            # Display the circuit
            print("+--- CIRCUIT ---+")
            print(self.drawer)

        @qml.qnode(self.device, interface="jax")
        def qcnn_circuit_prob(vqe_p, qcnn_p):
            self.vqe_qcnn_circuit(vqe_p, qcnn_p)

            if self.n_outputs == 1:
                return qml.probs(wires=self.N - 1)
            else:
                return [qml.probs(wires=int(k)) for k in self.final_active_wires]
        
        params = copy.copy(self.params)

        if inject:
            psi = []
            for h in self.vqe.Hs.mat_Hs:
                # Compute eigenvalues and eigenvectors
                eigval, eigvec = jnp.linalg.eigh(h)
                # Get the eigenstate to the lowest eigenvalue
                gstate = eigvec[:,jnp.argmin(eigval)]

                psi.append(gstate)
            psi = jnp.array(psi)
            self.psi = psi
            
            X_train = jnp.array(psi[train_index])
            test_index = np.setdiff1d(np.arange(len(psi)), train_index)
            X_test  = jnp.array(psi[test_index])

            @qml.qnode(self.device, interface="jax")
            def qcnn_circuit_prob(psi, qcnn_p):
                self.psi_qcnn_circuit(psi, qcnn_p)

                if self.n_outputs == 1:
                    return qml.probs(wires=self.N - 1)
                else:
                    return [qml.probs(wires=int(k)) for k in self.final_active_wires]
        
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
        self.circuit = self.vqe_qcnn_circuit if inject == False else self.psi_qcnn_circuit
        
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
