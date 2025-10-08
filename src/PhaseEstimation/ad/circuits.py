import numpy as np
import pennylane as qml


def demo(num_qubits, params, num_trashwires, depth):
    """Ansatz of the QAD model
    Apply multi-qubit gates between trash and non-trash wires
    """

    # Block of gates connecting trash and non-trash wires
    def block(nontrash, trash, shift):
        # Connect trash wires
        for i, wire in enumerate(trash):
            target = trash[(i + 1 + shift) % len(trash)]
            qml.CZ(wires=[int(wire), int(target)])
        # Connect each nontrash wire to a trash wire
        for i, wire in enumerate(nontrash):
            trash_idx = (i + shift) % len(trash)
            qml.CNOT(wires=[int(wire), int(trash[trash_idx])])

    # Define trash wires as a contiguous block in the middle.
    trash = np.arange(num_trashwires // 2, num_trashwires // 2 + num_trashwires)
    nontrash = np.setdiff1d(np.arange(num_qubits), trash)

    index = 0

    # Initial layer: apply RY rotations on all wires.
    for wire in np.arange(num_qubits):
        qml.RY(params[index], wires=int(wire))
        index += 1

    # Repeatedly apply blocks of entangling gates and additional rotations.
    for shift in range(depth):
        block(nontrash, trash, shift)
        qml.Barrier()
        # In the final layer, only apply rotations on trash wires.
        wires_to_rot = np.arange(num_qubits) if shift < depth - 1 else trash
        for wire in wires_to_rot:
            qml.RY(params[index], wires=int(wire))
            index += 1

    return index, list(trash)
