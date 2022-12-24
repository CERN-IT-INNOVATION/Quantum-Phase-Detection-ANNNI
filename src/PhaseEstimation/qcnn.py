""" This module implements the base functions to implement a Quantum Convolutional Neural Network (QCNN) for the (ANNNI) Ising Model. """
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

from matplotlib import pyplot as plt

import copy, tqdm, pickle

from PhaseEstimation import (
    circuits,
    vqe,
    general as qmlgen,
    ising_chain as ising,
    annni_model as annni,
    visualization as qplt,
)

from typing import Tuple, List, Callable
from numbers import Number

##############


def qcnn_circuit(params: List[Number], N: int, n_outputs: int) -> Tuple[int, List[int]]:
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
    np.ndarray
        Array of indexes of not-measured wires (due to pooling)
    """

    # Wires that are not measured (through pooling)
    active_wires = np.arange(N)

    # Visual Separation VQE||QCNN
    qml.Barrier()
    qml.Barrier()

    # Index of the parameter vector
    index = 0

    # Iterate Convolution+Pooling until we only have a single wires
    index = circuits.wall_gate(active_wires, qml.RY, params, index)
    circuits.wall_cgate_serial(active_wires, qml.CNOT)
    while len(active_wires) > n_outputs:  # Repeat until the number of active wires
        # (non measured) is equal to n_outputs
        # Convolute
        index = circuits.convolution(active_wires, params, index)
        # Measure wires and apply rotations based on the measurement
        index, active_wires = circuits.pooling(active_wires, qml.RX, params, index)

        qml.Barrier()

    circuits.wall_cgate_serial(active_wires, qml.CNOT)
    index = circuits.wall_gate(active_wires, qml.RY, params, index)

    # Return the number of parameters
    return index + 1, active_wires


class qcnn:
    def __init__(self, vqe: vqe.vqe, qcnn_circuit: Callable, n_outputs: int = 1):
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
        self.N = vqe.Hs.N
        self.n_states = vqe.Hs.n_states
        self.n_outputs = n_outputs
        self.qcnn_circuit_fun = lambda p: qcnn_circuit(p, self.N, n_outputs)
        self.n_params, self.final_active_wires = self.qcnn_circuit_fun([0] * 10000)
        self.params = np.array(np.random.rand(self.n_params))
        self.device = vqe.device

        self.vqe_params = np.array(vqe.vqe_params0)

        self.labels = np.array(vqe.Hs.labels)
        self.loss_train: List[float] = []
        self.loss_test: List[float] = []

    def __repr__(self):
        @qml.qnode(self.device, interface="jax")
        def circuit_drawer(self):
            _ = self.qcnn_circuit_fun(np.arange(self.n_params))
            if self.n_outputs == 1:
                return qml.probs(wires=self.N - 1)
            else:
                return qml.probs([int(k) for k in self.final_active_wires])

        return qml.draw(circuit_drawer)(self)

    def _vqe_qcnn_circuit(self, vqe_p, qcnn_p):
        """
        Circuit:
        VQE + QCNN
        """
        self.vqe.circuit(vqe_p)
        self.qcnn_circuit_fun(qcnn_p)

    # Training function
    def train(
        self,
        lr: float,
        n_epochs: int,
        train_index: List[Number],
        loss_fn: Callable,
        circuit: bool = False,
        plot: bool = False,
    ):
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
        """

        # -1 could be in the labels as [-1, -1] when training
        # ANNNI model which non-trivial cases have no solution
        if (-1 not in self.labels) and (None not in self.labels):
            X_train, Y_train = (
                jnp.array(self.vqe_params[train_index]),
                jnp.array(self.labels[train_index]),
            )
            test_index = np.setdiff1d(np.arange(len(self.vqe_params)), train_index)
            X_test, Y_test = (
                jnp.array(self.vqe_params[test_index]),
                jnp.array(self.labels[test_index]),
            )
        else:
            # If we are traing an ANNNI model, we have to first restrict on the trivial cases:
            # L = 0, K = whatever
            # K = 0, L = whatever
            mask = jnp.array(
                jnp.logical_or(
                    jnp.array(self.vqe.Hs.model_params)[:, 1] == 0,
                    jnp.array(self.vqe.Hs.model_params)[:, 2] == 0,
                )
            )

            self.vqe_params = jnp.array(self.vqe_params)

            X, Y = self.vqe_params[mask], self.labels[mask, :].astype(int)
            # The labels stored in the Hamiltonian class are:
            #   > [ 1, 1] for paramagnetic states
            #   > [ 0, 1] for ferromagnetic states
            #   > [ 1, 0] for antiphase states
            #   > [ 0, 0] not used
            #   > [-1,-1] for states with no analytical solutions
            # qml.probs(wires = active_wires) will output the following probabilities:
            # (example for a two qbits output)
            # p(00), p(01), p(10), p(11)
            # The labels need to be transformed accordingly
            #     [0,0] -> [1,0,0,0] trash case
            #     [0,1] -> [0,1,0,0] for ferromagnetic
            #     [1,0] -> [0,0,1,0] for antiphase
            #     [1,1] -> [0,0,0,1] for paramagnetic
            Ymix = []
            for label in Y:
                if (label == [0, 0]).all():
                    Ymix.append([1, 0, 0, 0])  # Trash
                elif (label == [0, 1]).all():
                    Ymix.append([0, 1, 0, 0])  # Ferromagnetic
                elif (label == [1, 0]).all():
                    Ymix.append([0, 0, 1, 0])  # Antiphase
                elif (label == [1, 1]).all():
                    Ymix.append([0, 0, 0, 1])  # Paramagnetic
            Y = jnp.array(Ymix)

            # The indexes of test are
            # All indexes (only analitical) \ train_index
            test_index = np.setdiff1d(np.arange(len(Y)), train_index)

            X_train, Y_train = X[train_index], Y[train_index]
            X_test, Y_test = X[test_index], Y[test_index]

        if circuit:
            # Display the circuit
            print("+--- CIRCUIT ---+")
            print(self)

        # QCircuit: Circuit(VQE, QCNNparams) -> probs
        @qml.qnode(self.device, interface="jax")
        def qcnn_circuit_prob(vqe_p, qcnn_p):
            self._vqe_qcnn_circuit(vqe_p, qcnn_p)

            return qml.probs([int(k) for k in self.final_active_wires])

        params = copy.copy(self.params)

        # Gradient of the Loss function
        jd_loss_fn = jax.jit(
            jax.grad(lambda p: loss_fn(X_train, Y_train, p, qcnn_circuit_prob))
        )

        # Update function
        # Returns updated parameters, updated state of the optimizer
        def update(params, opt_state):
            grads = jd_loss_fn(params)
            opt_state = opt_update(0, grads, opt_state)

            return get_params(opt_state), opt_state

        # Definying following function:
        # jitted loss function for training set loss(params)
        train_loss_fn = jax.jit(
            lambda p: loss_fn(X_train, Y_train, p, qcnn_circuit_prob)
        )
        # jitted loss function for test set loss(params)
        test_loss_fn = jax.jit(lambda p: loss_fn(X_test, Y_test, p, qcnn_circuit_prob))

        # Initialize tqdm progress bar
        progress = tqdm.tqdm(range(n_epochs), position=0, leave=True)

        # Defining an optimizer in Jax
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(params)

        loss_history, loss_history_test = [], []
        # Training loop:
        for epoch in range(n_epochs):
            params, opt_state = update(params, opt_state)

            # Every 100 iterations append the updated training (and testing) loss
            if epoch % 100 == 0:
                loss_history.append(train_loss_fn(params))
                if len(Y_test) > 0:
                    loss_history_test.append(test_loss_fn(params))

            # Update progress bar
            progress.update(1)
            progress.set_description("Cost: {0}".format(loss_history[-1]))

        # Update qcnn class after training
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

    def predict(self):
        """
        Get the phases probabilities for each VQE state

        Returns
        -------
        List[List[Number]]
            List of probabilities
        """

        @qml.qnode(self.device, interface="jax")
        def qcnn_circuit_prob(params_vqe, params):
            self._vqe_qcnn_circuit(params_vqe, params)

            return qml.probs([int(k) for k in self.final_active_wires])

        vcircuit = jax.vmap(lambda v: qcnn_circuit_prob(v, self.params), in_axes=(0))

        predictions = np.array(vcircuit(self.vqe_params))

        return predictions

    def predict_lines(self, predictions=[]):
        """
        Get the prdicted phase-transition line

        Parameters
        ----------
        predictions : List[List[Number]]
            This is the output of self.predict(), if it is not passed, the predictions will be computed asnew

        Returns
        -------
        List[Number]
            y-coordinate of the transition point for each kappa value
        """
        sidex, sidey = self.vqe.Hs.n_kappas, self.vqe.Hs.n_hs
        print(sidex, sidey)
        if len(predictions) == 0:
            predictions = self.predict()

        predictions = np.reshape(np.argmax(predictions, axis=1), (sidex, sidey))
        line_trans = []

        for col in range(sidex):
            y_cord_trans = 0
            for row in range(sidey - 1, -1, -1):
                prediction = predictions[col, row]
                if prediction != 3:
                    break
                y_cord_trans += 1

            line_trans.append(y_cord_trans)

        return np.array(line_trans)

    def save(self, filename: str):
        """
        Saves QCNN parameters to file

        Parameters
        ----------
        filename : str
            File where to save the parameters
        """
        if isinstance(filename, str):
            things_to_save = [self.params, self.qcnn_circuit_fun]

            with open(filename, "wb") as f:
                pickle.dump(things_to_save, f)
        else:
            raise TypeError("Invalid name for file")

    def show(self, train_index=[], marginal=False, **kwargs):
        if self.vqe.Hs.func == ising.build_Hs:
            qplt.QCNN_classification_ising(self, train_index)
        elif self.vqe.Hs.func == annni.build_Hs:
            if marginal:
                qplt.QCNN_classification_ANNNI_marginal(self)
            qplt.QCNN_classification_ANNNI(self, **kwargs)


def load(filename_vqe: str, filename_qcnn: str) -> qcnn:
    """
    Load QCNN from VQE file and QCNN file
    
    Parameters
    ----------
    filename_vqe : str
        Name of the file from where to load the VQE class
    filename_qcnn : str
        Name of the file from where to load the main parameters of the QCNN class
        
    Returns
    -------
    class
        QCNN class
    """
    if isinstance(filename_vqe, str) and isinstance(filename_qcnn, str):
        loaded_vqe = vqe.load_vqe(filename_vqe)

        with open(filename_qcnn, "rb") as f:
            params, qcnn_circuit_fun = pickle.load(f)

        loaded_qcnn = qcnn(loaded_vqe, qcnn_circuit_fun)
        loaded_qcnn.params = params

        return loaded_qcnn

    raise TypeError("Invalid name for file")


def get_trainset_gaussian(vqeclass: vqe.vqe, nS: int, sigma: float = 1) -> List[int]:
    """
    Draw randomly samples from the training for each axis according to the gaussian distribution
    centered around the phase transition on the axis and std sigma

    Parameters
    ----------
    vqeclass : vqe.vqe
        VQE class to get the side size of the system
    nS : int
        Number of samples to draw in total
    sigma : float
        Standard deviation of the two distributions

    Returns
    -------
    np.ndarray
        List of the indexes of the subset of the training set
    """
    side = vqeclass.Hs.side
    if nS > 2 * side - 1:
        raise ValueError("Subset size too large!")
    nS = nS // 2  # Size of the subset -> Number of samples to draw among each axis
    mu = side // 2  # Mean of the distributions

    training_set: List[int] = []
    # Get Y training set:
    while len(training_set) < nS:
        sample = int(
            np.random.normal(mu, sigma)
        )  # Draw randomly according to the gaussian distribution
        if sample not in training_set:  # No duplicates allowed
            if sample >= 0 and sample < side:  # Check if the drawn sample is in range
                training_set.append(sample)
    # Get X training set:
    while len(training_set) < 2 * nS:
        sample = (
            int(np.random.normal(mu, sigma)) + side
        )  # Draw randomly according to the gaussian distribution (and shift to the X axis)
        if sample not in training_set:  # No duplicates allowed
            if (
                sample >= side and sample < 2 * side
            ):  # Check if the drawn sample is in range
                training_set.append(sample)

    return np.array(training_set)


def ANNNI_accuracy(qcnnclass: qcnn, plot: bool = False) -> float:
    """
    Compute accuracy of the QCNN of the whole ANNNI state space

    Parameters
    ----------
    qcnnclass : qcnn
        QCNN class
    plot : bool
        if True -> displays the plot of the accuracy:
                if green: sample correctly classified
                if red  : sample wrongly classified
    
    Returns
    -------
    float
        Accuracy : (# samples correctly classified)/(# samples) (0,1)
    """
    circuit = qcnnclass._vqe_qcnn_circuit
    side = qcnnclass.vqe.Hs.side

    @qml.qnode(qcnnclass.device, interface="jax")
    def qcnn_circuit_prob(params_vqe, params):
        circuit(params_vqe, params)

        return [qml.probs(wires=int(k)) for k in qcnnclass.final_active_wires]

    vcircuit = jax.vmap(lambda v: qcnn_circuit_prob(v, qcnnclass.params), in_axes=(0))

    # Get the predictions of the QCNN among all states of the VQE
    predictions = np.array(np.argmax(vcircuit(qcnnclass.vqe_params), axis=2))

    # Compare predictions to actual states
    # applying inequalities to theoretical curves
    labels = []
    for idx in range(qcnnclass.vqe.Hs.n_states):
        # compute coordinates and normalize for x in [0,1]
        # and y in [0,2]
        x = (idx // side) / side
        y = 2 * (idx % side) / side

        # If x==0 we get into 0/0 on the theoretical curve
        if x == 0:
            if 1 <= y:
                labels.append([1, 1])
            else:
                labels.append([0, 1])
        elif x <= 0.5:
            if qmlgen.paraferro(x) <= y:
                labels.append([1, 1])
            else:
                labels.append([0, 1])
        else:
            if (qmlgen.paraanti(x)) <= y:
                labels.append([1, 1])
            else:
                labels.append([1, 0])

    correct = np.sum(np.array(labels) == predictions, axis=1).astype(int) == 2
    accuracy = np.sum(correct) / (side * side)

    if plot:
        plt.imshow(np.rot90(np.reshape(correct, (side, side))), cmap="RdYlGn")
        plt.show()

    return accuracy
