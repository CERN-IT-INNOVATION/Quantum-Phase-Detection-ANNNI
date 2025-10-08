import pickle
from pathlib import Path

import numpy as np
import pennylane as qml
import quimb as qu

from ..mps.convert import mps_to_qc
from .base import Input
from .utils import load_dict


class Brick(Input):
    def __init__(self, path, label=""):
        # Parse input dictionary and extract info
        brick_dict = load_dict(path, label)
        self.input_dict = {
            key: self.jnparrayze(brick_dict[key])[1] for key in brick_dict
        }

        keys = list(self.input_dict.keys())
        num_qubits = brick_dict[keys[0]][0][0][0][-1] + 1

        ks = [k for (k, _) in keys]
        hs = [h for (_, h) in keys]

        # Initialize from base class
        super().__init__(num_qubits=num_qubits, ks=ks, hs=hs)

        # Precompute wire pairs for ansatz
        self.pp_wire = [
            (max(0, i), max(1, i + 1)) for i in range(-1, self.num_qubits - 1)
        ][::-1]

    def ansatz(self, input_obj):
        for p_U in input_obj:
            for p_wire, U in zip(self.pp_wire, p_U):
                qml.QubitUnitary(U, wires=p_wire)

    def jnparrayze(self, conv_elem):
        def expand_unitary_to_two_qubits(U, wires):
            assert len(wires) == 2
            a, b = wires
            I = np.eye(2)
            return (np.kron(U, I) if a < b else np.kron(I, U)), [a, b]

        jnp_pp_wire, jnp_pp_U = [], []

        for p_wire, p_U in zip(conv_elem[0], conv_elem[1]):
            jnp_p_wire, jnp_p_U = [], []
            for wire, U in zip(p_wire, p_U):
                if len(wire) == 1:
                    padded_U, padded_wire = expand_unitary_to_two_qubits(
                        U, [wire[0], 1]
                    )
                else:
                    padded_U, padded_wire = U, wire
                jnp_p_wire.append(padded_wire)
                jnp_p_U.append(padded_U)
            jnp_pp_wire.append(jnp_p_wire)
            jnp_pp_U.append(jnp_p_U)

        return (jnp_pp_wire, jnp_pp_U)

    @staticmethod
    def make(k, h, mps, depth, path, label=""):
        filename = (
            Path(path) / f"{label}_k{k}_h{h}.pkl"
            if label
            else Path(path) / f"k{k}_h{h}.pkl"
        )

        mps_raw = qu.tensor.tensor_1d.MatrixProductState(mps, shape="lpr")
        _, _, pp_wire, pp_U = mps_to_qc(mps_raw, depth=depth)
        result = (pp_wire, pp_U)

        with open(filename, "wb") as f:
            pickle.dump(result, f)
