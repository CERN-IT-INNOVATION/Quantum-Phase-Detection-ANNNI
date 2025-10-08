import numpy as np
import quimb as qu
from quimb.tensor.tensor_1d import MatrixProductState


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
        tn_padded = _pad_tensor(
            tn, chi=chi, leftmost=(idx == 0), rightmost=(idx == mps.nsites - 1)
        )
        p_mps_padded_array.append(tn_padded)

    mps_padded = qu.tensor.tensor_1d.MatrixProductState(p_mps_padded_array, shape="lrp")
    return mps_padded


def gram_schmidt(matrix):
    unitary = np.zeros(matrix.shape, dtype=matrix.dtype)

    p_orthonormal_vec = []
    for i in range(matrix.shape[1]):
        col = matrix[:, i].copy()
        if not np.allclose(col, np.zeros(len(col))):
            p_orthonormal_vec += [col]

    for j in range(matrix.shape[1]):
        basis = np.array(p_orthonormal_vec)
        col = matrix[:, j].copy()
        if np.allclose(col, np.zeros(len(col))):
            real = np.random.uniform(-1, 1, len(col))
            imag = np.random.uniform(-1, 1, len(col)) if np.iscomplexobj(unitary) else 0
            col = real + 1j * imag

            for vec in basis:
                col -= (vec.conj().T @ col) * vec / np.linalg.norm(vec)

        if np.iscomplexobj(col):
            max_imag = np.max(np.abs(np.imag(col)))
            if max_imag > 1e-6:  # threshold for significant imaginary parts
                print(
                    f"Warning: column {j} has significant imaginary part (max |Im| = {max_imag})"
                )
            # Drop tiny imaginary parts safely
            col = np.real_if_close(col, tol=1e-10)

        unitary[:, j] = col / np.linalg.norm(col)
        p_orthonormal_vec += [unitary[:, j]]

    return unitary
