import numpy as np
import pennylane as qml
from quimb.tensor.tensor_1d import MatrixProductState

from PhaseEstimation.mps.utils import gram_schmidt, pad_mps


def mps_to_qc(mps: MatrixProductState, depth: None | int = None):
    def _chi2(mps):

        # Some MPS might have bond dimension 1 that breaks the algorithm
        # we pad the MPS to bond dimension 2 (at least)
        mps_can = pad_mps(mps.left_canonicalize(), 2)

        p_U = []  # List of numpy.array, the unitaries of the quantum circuit
        p_wire = []  # List of int, the wire(s) the unitaries are applied to
        p_gate = []  # List of pennylane.Gate, the gates of the quantum circuit

        for idx, tensor in reversed(list(enumerate(mps_can))):
            _tensor = np.swapaxes(tensor.data, 1, 2)
            ind_left, ind_p, ind_right = _tensor.shape
            isometry = _tensor.reshape((ind_p * ind_left, ind_right))

            matrix = np.zeros((isometry.shape[0], isometry.shape[0]))
            matrix[:, : isometry.shape[1]] = isometry

            # Compute the QR decomposition of the matrix
            U = gram_schmidt(matrix)

            # Spin(s)/Wire(s) the Unitary is applied to
            wire = [idx - 1, idx] if idx != 0 else [idx]

            # Decompose the unitary into individual gates
            for gate in qml.QubitUnitary(U, wires=wire).decomposition():
                p_gate.append(gate)

            p_wire.append(wire)
            p_U.append(U)

        return p_U, p_wire, p_gate

    mps_pad = pad_mps(mps, chi=mps.max_bond())
    mps_can = mps_pad.left_canonicalize()
    depth = np.log2(mps.max_bond()).astype(int) if depth is None else depth

    mps_temp = mps_can.copy(deep=True)
    pp_U, pp_wire, pp_gate = [], [], []
    for layer in range(depth):

        mps_can_compressed = mps_temp.copy(deep=True)
        mps_can_compressed.compress(form="left", max_bond=2)
        mps_can_compressed.normalize()

        p_U, p_wire, p_gate = _chi2(mps_can_compressed)
        pp_U.append(p_U)
        pp_wire.append(p_wire)
        pp_gate.append(p_gate)

        # Apply the inverse of U_k to disentangle |ψ_k>,
        # |ψ_(k+1)> = inv(U_k) @ |ψ_k>.
        for i in range(len(p_U)):
            inverse = p_U[-(i + 1)].conj().T
            if inverse.shape[0] == 4:
                mps_temp.gate_split(inverse, (i - 1, i), inplace=True, cutoff=0)
            else:
                mps_temp.gate(inverse, (i), inplace=True, contract=True)

    def circuit():
        [qml.apply(gate) for p_gate in pp_gate[::-1] for gate in p_gate]

    def circuitU():
        for p_wire, p_U in zip(pp_wire[::-1], pp_U[::-1]):
            for wire, U in zip(p_wire, p_U):
                qml.QubitUnitary(U, wires=wire)

    return circuit, circuitU, pp_wire[::-1], pp_U[::-1]
