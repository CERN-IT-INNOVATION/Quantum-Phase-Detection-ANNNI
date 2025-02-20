from typing import Tuple, List
import pennylane as qml
import numpy as np

def ID9(n_qubit : int, p_v: np.ndarray, **kwargs) -> int:
    """
    Adapted ansatz from the paper:
    Expressibility and entangling capability of parameterized quantum circuits for 
    hybrid quantum-classical algorithms (https://arxiv.org/abs/1905.10876
    
    Parameters
    ----------
    n_qubit : int
        Number of qubit of the ansatz
    p_v : np.ndarray
        Array of the trainable parameters of the VQE

    Returns
    -------
    int
        Number of trainable parameters
    """
    def block(n_qubit: int, p_v: np.ndarray, index: int):
        for qubit in range(n_qubit):
            qml.Hadamard(wires=[qubit])
        for qubit_c, qubit_t in zip(range(n_qubit), range(n_qubit)[1:]):
            qml.CNOT(wires=[qubit_c, qubit_t])
        for qubit in range(n_qubit):
            qml.RY(p_v[index], wires=[qubit])
            index += 1

        return index

    n_iteration = kwargs.get('n_iteration', 1)

    index_v = 0
    for depth in range(n_iteration):
        index_v = block(n_qubit, p_v, index_v)
        qml.Barrier()

    return index_v

def anomaly(n_qubit, p_p) -> Tuple[int, List[int]]:
    """
    Returns
    -------
    int
        Total number of parameters needed to build this circuit
    np.ndarray
        Array of the trainable parameters of the Anomaly Detection model
    """

    def block(wires: List[int], wires_trash: List[int], shift: int = 0):
        # Connection between trash wires
        trash_uniques = []
        for wire in wires_trash:
            wire_target = wire + 1 + shift

            if wire_target > wires_trash[-1]:
                wire_target = wires_trash[0] + wire_target - wires_trash[-1] - 1
            if wire_target == wire:
                wire_target += 1
            if wire_target > wires_trash[-1]:
                wire_target = wires_trash[0] + wire_target - wires_trash[-1] - 1

            if [wire_target, wire] not in trash_uniques:
                qml.CZ(wires=[int(wire), int(wire_target)])
                trash_uniques.append([wire, wire_target])

        # Connections wires -> trash_wires
        for idx, wire in enumerate(wires):
            trash_idx = idx + shift

            while trash_idx > len(wires_trash) - 1:
                trash_idx = trash_idx - len(wires_trash)

            qml.CNOT(wires=[int(wire), int(wires_trash[trash_idx])])
    
    def wall(p_wire, rot_fun, p_p, index):
        for wire in p_wire:
            rot_fun(p_p[index], wires=int(wire))
            index += 1
        return index
    
    depth = 3

    # Number of wires that will not be measured
    n_trashwire = n_qubit // 2

    # Constructing the array for the measured qubits (from the middle part)
    p_trashwire = np.arange(n_trashwire // 2, n_trashwire // 2 + n_trashwire)

    p_nontrashwire = np.setdiff1d(np.arange(n_qubit), p_trashwire)

    # Index of the parameter vector
    index = 0

    index = wall(np.arange(n_qubit), qml.RY, p_p, index)
    
    for shift in range(depth):
        block(p_nontrashwire, p_trashwire, shift)
        qml.Barrier()
        p_wire = np.arange(n_qubit) if shift < depth - 1 else p_trashwire
        index = wall(p_wire, qml.RY, p_p, index)
        # index = wall(p_wire, qml.RX, p_p, index)

    # Return the number of parameters
    return index, p_trashwire

def qcnn(n_qubit, p_p) -> Tuple[int, List[int]]:
    """
    Returns
    -------
    int
        Total number of parameters needed to build this circuit
    np.ndarray
        Array of the trainable parameters of the QCNN model
    """
    
    def wall(p_wire, rot_fun, p_p, index):
        for wire in p_wire:
            rot_fun(p_p[index], wires=int(wire))
            index += 1
        return index
    
    def conv(p_wire, p_p, index):
        """Apply convolution layer over groups of 2 or 4 wires."""
        p_group = []
        if len(p_wire) % 4 == 0:
            p_group = p_wire.reshape(-1, 4)
        elif len(p_wire) % 2 == 0:
            p_group = p_wire.reshape(-1, 2)
        else:
            p_group = p_wire[:-1].reshape(-1, 2)
            qml.RY(p_p[index], wires=int(p_wire[-1]))
            index += 1
            
        for p_wire_group in p_group:
            for wire1, wire2 in zip(p_wire_group[0::1], p_wire_group[1::1]):
                qml.CNOT(wires=[int(wire1), int(wire2)])
            for wire in p_wire_group:
                qml.RY(p_p[index], wires=int(wire))
                index += 1
            
        return index

    def pool(p_wire, p_p, index):
        is_even = len(p_wire) % 2 == 0

        for wire_meas, wire_next in zip(p_wire[0::2], p_wire[1::2]):
            m_0 = qml.measure(int(wire_meas))
            qml.cond(m_0 == 0, qml.RX)(p_p[index],     wires=int(wire_next))
            qml.cond(m_0 == 1, qml.RX)(p_p[index + 1], wires=int(wire_next))
            index = index + 2

            # Removing measured wires from active_wires:
            p_wire = np.delete(p_wire, np.where(p_wire == wire_meas))

        # ---- > If the number of wires is odd, the last wires is not pooled
        #        so we apply a gate
        if not is_even:
            qml.RX(p_p[index], wires=int(p_wire[-1]))
            index = index + 1

        return index, p_wire

    # Wires that are not measured (through pooling)
    p_activewire = np.arange(n_qubit)

    # Index of the parameter vector
    index = 0
    output_dim = 2

    index = wall(p_activewire, qml.RY, p_p, index)

    while len(p_activewire) > output_dim:
        # Convolute
        index = conv(p_activewire, p_p, index)
        # Pool
        index, p_activewire = pool(p_activewire, p_p, index)
        
        qml.Barrier()

    # index = conv(p_activewire, p_p, index)
    index = wall(p_activewire, qml.RY, p_p, index)

    # Return the number of parameters
    return index, p_activewire