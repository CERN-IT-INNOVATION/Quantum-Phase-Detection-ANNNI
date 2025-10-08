"""This module implements the base functions to implement a Quantum Convolutional Neural Network (QCNN) for the (ANNNI) Ising Model."""

import pennylane as qml
from jax import config, jit, vmap
from pennylane import numpy as np

from . import circuits

# Training raises UserWarning:
# Explicitly requested dtype <class 'jax.numpy.complex128'>
# requested in astype is not available, and will be truncated
# to dtype complex64.
# TODO: Is complex64 sufficient? Does it bring more speed?
config.update("jax_enable_x64", True)


class Qcnn:
    def __init__(
        self,
        Input,
        ansatz: str = "demo",
    ):
        self.Input = Input
        self.ansatz_name = ansatz
        self.block_fun = getattr(circuits, ansatz)

        self.num_qubits = self.Input.num_qubits
        self.side = self.Input.side
        self.hs = self.Input.hs
        self.ks = self.Input.ks

        self.device = qml.device("default.qubit", wires=self.num_qubits, shots=None)

        # Number of parameters | List of the output wires
        self.num_params, self.wires_output = self.block_fun(
            self.num_qubits, np.arange(10000)
        )
        # Random initial parameter vector
        self.params = np.random.normal(loc=0, scale=1, size=self.num_params)

        self.hs = np.linspace(0, 2, self.side)
        self.ks = np.linspace(0, 1, self.side)

        self.q_circuit = qml.QNode(self.circuit, device=self.device)
        self.jq_circuit = jit(self.q_circuit)
        self.vjq_circuit = vmap(self.jq_circuit, in_axes=(None, 0))

        self.states, self.labels, self.analytical_mask, self.index_map = (
            self.Input.load_all()
        )

    def ansatz(self, params, input_obj):
        self.Input.ansatz(input_obj)

        qml.Barrier()
        qml.Barrier()

        self.block_fun(self.num_qubits, params)

    def circuit(self, params, input_obj):
        self.ansatz(params=params, input_obj=input_obj)
        return qml.probs([int(k) for k in self.wires_output])

    def predict(self, params, sequential=False):
        if sequential:
            Ypred = [self.jq_circuit(params, X) for X in self.states]
        else:
            Ypred = self.vjq_circuit(params, self.states)
        return np.flip(np.rot90(np.reshape(Ypred, (self.side, self.side, -1)), -1), 1)

    def show(self, mpl=False):
        if mpl:
            qml.draw_mpl(self.q_circuit)(
                np.arange(self.num_params), np.array(self.states[0])
            )
        else:
            print(
                qml.draw(self.q_circuit)(
                    np.arange(self.num_params), np.array(self.states[0])
                )
            )

    def __repr__(self):
        repr_str = "QCNN Class:"
        repr_str += f"\n  num_qubits : {self.num_qubits}"
        repr_str += f"\n  side       : {self.side}"
        repr_str += f"\n  num_params : {self.num_params}"

        return repr_str
