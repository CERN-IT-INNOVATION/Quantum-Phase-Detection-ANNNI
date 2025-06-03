import numpy as np
import quimb as qu
import pennylane as qml
from quimb.tensor.tensor_1d import MatrixProductState

from PhaseEstimation.utils.math import gram_schmidt

def _pad_tensor(tensor, chi, leftmost=False, rightmost=False):
    Dl, Dr, Dp = tensor.data.shape

    Dl_new = chi if chi >= Dl else Dl
    Dr_new = chi if chi >= Dr else Dr

    Dl_new, Dr_new, Dp_new = chi, chi, Dp
    if leftmost and Dl == 1:
        Dl_new = Dl
    if rightmost and Dr == 1:
        Dr_new = Dr

    assert Dl_new >= Dl and Dr_new >= Dr and Dp_new >= Dp

    # Initialize padded tensor with zeros
    padded = np.zeros((Dl_new, Dr_new, Dp_new), dtype=tensor.data.dtype)

    # Copy original data
    padded[:Dl, :Dr, :Dp] = tensor.data

    # Add identity block to new bonds, physical index = 0 (or could be 1)
    for i in range(Dl, Dl_new):
        for j in range(Dr, Dr_new):
            if i == j:
                padded[i, j, 0] = 1.0  # Insert identity in phys=0 subspace

    return padded


def pad_mps(mps: MatrixProductState, chi: int) -> MatrixProductState:
    """
    Pads a matrix product state to the target bond dimension.
    
    Parameters:
    - mps: quimb.tensor.tensor_1d.MatrixProductState
    - chi: int, target bond dimension

    Returns:
    - mps_padded: quimb.tensor.tensor_1d.MatrixProductState
    """
    p_mps_padded_array = []
    for idx, tn in enumerate(mps):
        tn_padded = _pad_tensor(tn, chi = chi, leftmost = (idx == 0), rightmost = (idx == mps.nsites - 1))
        p_mps_padded_array.append(tn_padded)

    mps_padded = qu.tensor.tensor_1d.MatrixProductState(p_mps_padded_array, shape='lrp')
    return mps_padded

def mps_to_qc(mps : MatrixProductState, depth : None | int = None):
    """
    Encodes a Matrix Product State (MPS) into a quantum circuit.
    
    Parameters:
    - mps: quimb.tensor.tensor_1d.MatrixProductState
    - depth: int (optional), number of layers in the quantum circuit. If None, it is set to the maximum bond dimension of the MPS.
    
    Returns:
    - circuit: callable, the quantum circuit
    - circuitU: callable, the quantum circuit with the unitaries (for testing purposes)
    """

    def _chi2(mps):
        """
        Encodes a Matrix Product State (MPS) of bond dimension 2 into a quantum circuit.
        
        Parameters:
        - mps: quimb.tensor.tensor_1d.MatrixProductState
        
        Returns:
        - p_U: list of numpy.array, the unitaries of the quantum circuit
        - p_wire: list of int, the wire(s) the unitaries are applied to
        - p_gate: list of pennylane.Gate, the gates of the quantum circuit
        """
        
        # Some MPS might have bond dimension 1 that breaks the algorithm
        # we pad the MPS to bond dimension 2 (at least)
        mps_can = pad_mps(mps.left_canonicalize(), 2)

        p_U    = []  # List of numpy.array, the unitaries of the quantum circuit
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

    mps_pad = pad_mps(mps, chi = mps.max_bond()) 
    mps_can = mps_pad.left_canonicalize()
    depth = np.log2(mps.max_bond()).astype(int) if depth is None else depth
    
    mps_temp = mps_can.copy(deep = True)
    pp_U, pp_wire, pp_gate = [], [], []
    for layer in range(depth):

        mps_can_compressed = mps_temp.copy(deep = True)
        mps_can_compressed.compress(form = "left", max_bond = 2)
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
                mps_temp.gate_split(
                    inverse, (i - 1, i), inplace=True, cutoff=0
                )
            else:
                mps_temp.gate(inverse, (i), inplace=True, contract=True)

    def circuit():
        [qml.apply(gate) for p_gate in pp_gate[::-1] for gate in p_gate]

    def circuitU():
        for p_wire, p_U in zip(pp_wire[::-1], pp_U[::-1]):
            for wire, U in zip(p_wire, p_U):
                qml.QubitUnitary(U, wires=wire)
    
    return circuit, circuitU
