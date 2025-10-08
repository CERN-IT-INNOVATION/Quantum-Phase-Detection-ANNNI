import pickle
from pathlib import Path

import numpy as np
import pennylane as qml

from ..annni.hamiltonian import get_energy_psi
from .base import Input
from .utils import load_dict


class Diag(Input):
    def __init__(self, path, label=""):
        # Parse input dictionary and extract info
        self.input_dict = load_dict(path, label)

        keys = list(self.input_dict.keys())
        ks = [k for (k, _) in keys]
        hs = [h for (_, h) in keys]

        num_qubits = int(np.log2(len(self.input_dict[keys[0]])))

        # Initialize from base class
        super().__init__(num_qubits=num_qubits, ks=ks, hs=hs)

    def ansatz(self, input_obj):
        qml.StatePrep(input_obj, wires=range(self.num_qubits), normalize=True)

    @staticmethod
    def make(k, h, num_qubits, path, label=""):
        filename = (
            Path(path) / f"{label}_k{k}_h{h}.pkl"
            if label
            else Path(path) / f"k{k}_h{h}.pkl"
        )
        with open(filename, "wb") as f:
            pickle.dump(get_energy_psi(num_qubits, k, h)[1], f)
