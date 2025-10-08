"""This module implements the base functions for the Quantum Anomaly Detection"""

import jax.numpy as jnp
import pennylane as qml
from jax import jit, vmap
from pennylane import numpy as np

from . import circuits


class Ad:
    def __init__(
        self,
        Input,
        ansatz: str = "demo",
        depth=1,
        num_trashwires=None,
    ):
        self.Input = Input
        self.ansatz_name = ansatz
        self.block_fun = getattr(circuits, ansatz)
        self.depth = depth
        self.num_qubits = self.Input.num_qubits
        self.side = self.Input.side
        self.hs = self.Input.hs
        self.ks = self.Input.ks

        if num_trashwires is None:
            self.num_trashwires = self.num_qubits // 2
        else:
            self.num_trashwires = num_trashwires

        self.device = qml.device("default.qubit", wires=self.num_qubits, shots=None)

        # Number of parameters | List of the output wires
        self.num_params, self.trashwires = self.block_fun(
            self.num_qubits, np.arange(10000), self.num_trashwires, self.depth
        )

        self.q_circuit = qml.QNode(self.circuit, device=self.device)
        self.jq_circuit = jit(self.q_circuit)
        self.vjq_circuit = vmap(self.jq_circuit, in_axes=(None, 0))

        self.states, _, _, self.index_map = self.Input.load_all()

    def ansatz(self, params, input_obj):
        self.Input.ansatz(input_obj)

        # Visual Separation VQE||QCNN
        qml.Barrier()
        qml.Barrier()

        self.block_fun(self.num_qubits, params, self.num_trashwires, self.depth)

    def circuit(self, params, input_obj):
        self.ansatz(params=params, input_obj=input_obj)
        return [qml.expval(qml.PauliZ(int(k))) for k in self.trashwires]

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

    def compress(self, params):
        """
        Output the compression score on the full phase space
        """
        Z = jnp.array([self.jq_circuit(params, state) for state in self.states])
        Zn = (1 + Z) / 2
        compression = jnp.mean(Zn, axis=1)
        compression = np.reshape(compression, (self.side, self.side))
        compression = np.flip(np.rot90(compression, 1), 0)

        return compression

    def v_compress(self, params):
        """
        Output the compression score on the full phase space
        """
        Z = jnp.array(self.vjq_circuit(params, self.states))
        Zn = (1 + Z) / 2
        compression = jnp.mean(Zn, axis=0)
        compression = np.reshape(compression, (self.side, self.side))
        compression = np.flip(np.rot90(compression, 1), 0)
        return compression

    def __repr__(self):
        repr_str = "Anomaly Detection Class:"
        repr_str += f"\n  num_qubits : {self.num_qubits}"
        repr_str += f"\n  side       : {self.side}"
        repr_str += f"\n  num_params : {self.num_params}"

        return repr_str
