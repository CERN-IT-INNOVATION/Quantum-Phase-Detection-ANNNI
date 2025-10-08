import pickle
from pathlib import Path

import numpy as np
import pennylane as qml
import quimb as qu

from ..annni.dmrg import dmrg_annni
from .base import Input
from .utils import load_dict


class Mps(Input):
    def __init__(self, path, label=""):
        # Parse input dictionary and extract info
        mps_dict = load_dict(path, label)
        self.input_dict = {key: self.mps_to_wf(mps_dict[key]) for key in mps_dict}

        keys = list(self.input_dict.keys())
        ks = [k for (k, _) in keys]
        hs = [h for (_, h) in keys]

        num_qubits = int(np.log2(len(self.input_dict[keys[0]])))

        # Initialize from base class
        super().__init__(num_qubits=num_qubits, ks=ks, hs=hs)

    def ansatz(self, input_obj):
        qml.StatePrep(input_obj, wires=range(self.num_qubits), normalize=True)

    def mps_to_wf(self, mps):
        return (
            qu.tensor.tensor_1d.MatrixProductState(mps, shape="lpr")
            .to_dense()
            .flatten()
        )

    @staticmethod
    def make(k, h, num_qubits, chi, path, label=""):
        filename = (
            Path(path) / f"{label}_k{k}_h{h}.pkl"
            if label
            else Path(path) / f"k{k}_h{h}.pkl"
        )
        with open(filename, "wb") as f:
            pickle.dump(dmrg_annni(num_qubits, chi, k, h)[1], f)
