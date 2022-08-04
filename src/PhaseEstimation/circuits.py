""" This module implements base circuits layout for all the models used"""
import pennylane as qml
from pennylane import numpy as np
##############

def wall_gate(active_wires, gate, params = [], index = 0, samerot = False):
    """
    Apply independent rotations of the same gate for all the wires (active_wires)
    
    Parameters
    ----------
    active_wires : np.ndarray
        Array of the wires to apply the rotations to
    gate : Pennylane gate (parametrized or not)
        Qubit operator to apply
    params : list
        List of parameters of the whole circuit
    index : int
        Starting index for this block of operators
    
    Returns
    -------
    int
        Updated index value
    """
    if len(params) > 0:
        if not samerot:
            for i, spin in enumerate(active_wires):
                gate(params[index + i], wires = int(spin) )
            return index + i + 1
        else:
            for spin in active_wires:
                gate(params[index], wires = int(spin) )
            return index + 1
    else:
        for spin in active_wires:
            gate(wires = int(spin) )
        return index
    
def wall_cgate_serial(active_wires, cgate, params = [], index = 0, going_down = True):
    """
    Apply drop-down controlled rotations for all the wires (active_wires)
    
    Parameters
    ----------
    active_wires : np.ndarray
        Array of the wires to apply the rotations to
    cgate : Pennylane gate (parametrized or not)
        Qubit controlled operator to apply
    params : list
        List of parameters of the whole circuit
    index : int
        Starting index for this block of operators
    going_down : bool
        if True -> top - down, if False -> down - top
    
    Returns
    -------
    int
        Updated index value
    """
    if len(params) > 0:
        for i, (spin, spin_next) in enumerate(zip(active_wires, active_wires[1:])):
            if going_down:
                cgate(params[index + i], wires=[int(spin), int(spin_next)] )
            else:
                cgate(params[index + i], wires=[int(spin_next), int(spin)] )
        return index + i + 1
    else:
        for spin, spin_next in zip(active_wires, active_wires[1:]):
            if going_down:
                cgate(wires=[int(spin), int(spin_next)] )
            else:
                cgate(wires=[int(spin_next), int(spin)] )
        return index
    
def wall_cgate_all(active_wires, cgate, params = [], index = 0, going_down = True):
    """
    Apply controlled rotations across all the wires (active_wires)
    
    Parameters
    ----------
    active_wires : np.ndarray
        Array of the wires to apply the rotations to
    cgate : Pennylane gate (parametrized or not)
        Qubit controlled operator to apply
    params : list
        List of parameters of the whole circuit
    index : int
        Starting index for this block of operators
    going_down : bool
        if True -> top - down, if False -> down - top
    
    Returns
    -------
    int
        Updated index value
    """
    if len(params) > 0:
        i = 0
        for k, spin in enumerate(active_wires):
            for spin_next in active_wires[k+1:]:
                if going_down:
                    cgate(params[index + i], wires=[int(spin), int(spin_next)] )
                else:
                    cgate(params[index + i], wires=[int(spin_next), int(spin)] )
                i += 1
        return index + i + 1
    else:
        for k, spin in enumerate(active_wires):
            for spin_next in active_wires[k+1:]:
                if going_down:
                    cgate(wires=[int(spin), int(spin_next)] )
                else:
                    cgate(wires=[int(spin_next), int(spin)] )
        return index

def wall_cgate_nextneighbour(active_wires, cgate, params = [], index = 0, going_down = True):
    """
    Apply drop-down controlled rotations establishing next-neighbour entanglement
    
    Parameters
    ----------
    active_wires : np.ndarray
        Array of the wires to apply the rotations to
    cgate : Pennylane gate (parametrized or not)
        Qubit controlled operator to apply
    params : list
        List of parameters of the whole circuit
    index : int
        Starting index for this block of operators
    going_down : bool
        if True -> top - down, if False -> down - top
    
    Returns
    -------
    int
        Updated index value
    """
    if len(params) > 0:
        for i, (spin, spin_next) in enumerate(zip(active_wires, active_wires[2:])):
            if going_down:
                cgate(params[index + i], wires=[int(spin), int(spin_next)] )
            else:
                cgate(params[index + i], wires=[int(spin_next), int(spin)] )
        return index + i + 1
    else:
        for spin, spin_next in zip(active_wires, active_wires[2:]):
            if going_down:
                cgate(wires=[int(spin), int(spin_next)] )
            else:
                cgate(wires=[int(spin_next), int(spin)] )
        return index

def circuit_ID9(active_wires, params, index = 0):
    """
    Basic block for VQE
    
    Parameters
    ----------
    active_wires : np.ndarray
        Array of the wires to apply the rotations to
    params : list
        List of parameters of the whole circuit
    index : int
        Starting index for this block of operators
        
    Returns
    -------
    int
        Updated index value
    """
    wall_gate(active_wires, qml.Hadamard)
    wall_cgate_serial(active_wires, qml.CNOT)
    index = wall_gate(active_wires, qml.RY, params, index)
    
    return index

def pooling(active_wires, qmlrot_func, params, index = 0):
    """
    Pooling block for the QCNN

    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured during a previous pooling
    qmlrot_func : function
        Pennylane Gate function to apply
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    np.ndarray
        Updated array of active wires (not measured)
    """
    for wire_meas, wire_next in zip(active_wires[0::2], active_wires[1::2]):
        m_0 = qml.measure(int(wire_meas))
        qml.cond(m_0 == 0, qmlrot_func)(params[index], wires=int(wire_next))
        qml.cond(m_0 == 1, qmlrot_func)(params[index + 1], wires=int(wire_next))
        index = index + 2

        # Removing measured wires from active_wires:
        active_wires = np.delete(active_wires, np.where(active_wires == wire_meas))

    # ---- > If the number of wires is odd, the last wires is not pooled
    #        so we apply a gate
    if len(active_wires) % 2 != 0:
        qmlrot_func(params[index], wires = int(active_wires[-1]) )
        index = index + 1
     
    #index = wall_gate(active_wires, qml.RX, params, index)
    
    return index, active_wires

def convolution(active_wires, params, index = 0):
    """
    Convolution block for the QCNN

    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured during a previous pooling
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    if len(active_wires) > 1:
         # Rotation Groups 2
        for wire1, wire2 in zip(active_wires[0::2],active_wires[1::2]):
            qml.RY(params[index], wires = int(wire1))
            qml.RY(params[index], wires = int(wire2))
            index += 1

        if len(active_wires)%2 != 0:
            qml.RY(params[index], wires = int(active_wires[-1]))
            index += 1
            
        # CNOTS Groups 1
        for wire1, wire2 in zip(active_wires[1::2],active_wires[2::2]):
            qml.CNOT(wires = [int(wire1),int(wire2)])
            
        qml.Barrier()
        
        # Rotation Groups 1
        qml.RY(params[index], wires = int(active_wires[0]))
        index += 1
        for wire1, wire2 in zip(active_wires[1::2],active_wires[2::2]):
            qml.RY(params[index], wires = int(wire1))
            qml.RY(params[index], wires = int(wire2))
            index += 1

        if len(active_wires)%2 == 0:
            qml.RY(params[index], wires = int(active_wires[-1]))
            index += 1
            
        # CNOTS Groups 2
        for wire1, wire2 in zip(active_wires[0::2],active_wires[1::2]):
            qml.CNOT(wires = [int(wire1),int(wire2)])
        
        index = wall_gate(active_wires, qml.RY, params, index)
        
    return index

def encoder_block(wires, wires_trash, shift = 0):
    """
    Applies CX between a wire and a trash wire for each
    wire/trashwire

    Parameters
    ----------
    wires : np.ndarray
        Array of the indexes of non-trash qubits
    wires_trash : np.ndarray
        Array of the indexes of trash qubits (np.1dsetdiff(np.arange(N),wires))
    shift : int
        Shift value for connections between wires and trash wires
    """
    # Connection between trash wires
    trash_uniques = []
    for wire in wires_trash:
        wire_target = wire + 1 + shift
        
        if wire_target > wires_trash[-1]:
            wire_target = wires_trash[0] + wire_target - wires_trash[-1] -1
        if wire_target == wire:
            wire_target += 1
        if wire_target > wires_trash[-1]:
            wire_target = wires_trash[0] + wire_target - wires_trash[-1] -1
            
        if not [wire_target, wire] in trash_uniques:
            qml.CNOT(wires=[int(wire), int(wire_target)])
            trash_uniques.append([wire,wire_target])

    # Connections wires -> trash_wires
    for idx, wire in enumerate(wires):
        trash_idx = idx + shift
        
        while trash_idx > len(wires_trash) - 1:
            trash_idx = trash_idx - len(wires_trash)
        
        qml.CNOT(wires=[int(wire), int(wires_trash[trash_idx])])

def encoder_circuit(wires, wires_trash, active_wires, params, index = 0):
    """
    Encoder circuit for encoder and autoencoder

    Parameters
    ----------
    wires : np.ndarray
        Array of the indexes of non-trash qubits
    wires_trash : np.ndarray
        Array of the indexes of trash qubits (np.1dsetdiff(np.arange(N),wires))
    active_wires : np.ndarray
        wires U wires_trash
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array
        
    Returns
    -------
    int
        Updated index value
    """
    for shift in range(len(wires_trash)):
        index = wall_gate(active_wires, qml.RY, params, index = index)
        encoder_block(wires, wires_trash, shift)
        qml.Barrier()
    index = wall_gate(active_wires, qml.RY, params, index = index)

    return index

def decoder_block(wires_trash, wires_extra, params, index, shift = 0):
    """
    (Almost) specular encoder block

    Parameters
    ----------
    wires_trash : np.ndarray
        Array of the indexes of trash qubits (np.1dsetdiff(np.arange(N),wires))
    wires_extra : np.ndarray
        New wires for decoding
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array
    shift : int
        Shift value for connections between wires and trash wires
        
    Returns
    -------
    int
        Updated index value
    """
    shift = shift % 2
    
    enc = False
    k = 0
    for wire_target in wires_extra[shift::shift+1]:
        qml.CRX(params[index + k], wires = [int(wires_trash[int(enc)]), int(wire_target)])
        k += 1        
        enc = not enc
    
    qml.CRY(params[index + k], wires = [int(wires_extra[shift::shift+1][-2]),int(wires_trash[int(not enc)])])
    k += 1
    qml.CRY(params[index + k], wires = [int(wires_extra[shift::shift+1][-1]),int(wires_trash[int(enc)])])
    
    return index + k + 1

def decoder_circuit(wires_trash, wires_extra, params, index = 0):
    """
    (Almost) specular encoder circuit for autoencoder

    Parameters
    ----------
    wires_trash : np.ndarray
        Array of the indexes of trash qubits (np.1dsetdiff(np.arange(N),wires))
    wires_extra : np.ndarray
        New wires for decoding
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array
        
    Returns
    -------
    int
        Updated index value
    """
    
    index = wall_gate(wires_extra, qml.RY, params, index = index)
    index = wall_gate(wires_extra, qml.RX, params, index = index)
    index = wall_gate(wires_extra, qml.RY, params, index = index)
        
    for shift in range(8):
        index = decoder_block(wires_trash, wires_extra, params, index, shift)
        qml.Barrier()
        
        index = wall_gate(wires_trash, qml.RX, params, index = index)
        index = wall_gate(wires_extra, qml.RX, params, index = index)
        
        qml.Barrier()

    for n, m in enumerate(wires_extra):
        qml.SWAP(wires = [int(n),int(m)])
    
    return index
    