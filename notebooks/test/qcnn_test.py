""" This module implements the base functions to implement a Quantum Convolutional Neural Network (QCNN) for the (ANNNI) Ising Model. """
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

from matplotlib import pyplot as plt

import copy, tqdm, pickle

from PhaseEstimation import circuits, vqe, general as qmlgen, ising_chain as ising, annni_model as annni, visualization as qplt

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
    def __init__(self, N : int, qcnn_circuit: Callable):
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
        self.N = N
        self.n_outputs = 2
        self.qcnn_circuit_fun = lambda p: qcnn_circuit(p, self.N, self.n_outputs)
        self.n_params, self.final_active_wires = self.qcnn_circuit_fun([0] * 10000)
        self.params = np.array(np.random.rand(self.n_params))
        self.device = qml.device("default.qubit.jax", wires=self.N, shots=None)

    def __repr__(self):
        @qml.qnode(self.device, interface="jax")
        def circuit_drawer(self):
            _ = self.qcnn_circuit_fun(np.arange(self.n_params))
            if self.n_outputs == 1:
                return qml.probs(wires=self.N - 1)
            else:
                return qml.probs([int(k) for k in self.final_active_wires])

        return qml.draw(circuit_drawer)(self)
    
    # Training function
    def train(
        self,
        n_epochs: int,
    ):

        lr = 1e-2

        # QCircuit: Circuit(VQE, QCNNparams) -> probs
        @qml.qnode(self.device, interface="jax")
        def qcnn_circuit_prob(qcnn_p):
            self.qcnn_circuit_fun(qcnn_p)

            return qml.probs([int(k) for k in self.final_active_wires])

        params = copy.copy(self.params)

        def loss_fn(params, q_circuit):
            qcnn_prob = lambda : q_circuit(params)

            predictions = 2 * qcnn_prob() - 1
            Y = 0

            hinge_loss = jnp.mean(1 - predictions * Y)

            return hinge_loss

        # Gradient of the Loss function
        jd_loss_fn = jax.jit(
            jax.grad(lambda p: loss_fn(p, qcnn_circuit_prob))
        )

        # Update function
        # Returns updated parameters, updated state of the optimizer
        def update(params, opt_state):
            grads = jd_loss_fn(params)
            opt_state = opt_update(0, grads, opt_state)

            return get_params(opt_state), opt_state

        # Initialize tqdm progress bar
        progress = tqdm.tqdm(range(n_epochs), position=0, leave=True)

        # Defining an optimizer in Jax
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(params)

        # Training loop:
        for epoch in range(n_epochs):
            params, opt_state = update(params, opt_state)

            # Update progress bar
            progress.update(1)

        # Update qcnn class after training
        self.params = params

