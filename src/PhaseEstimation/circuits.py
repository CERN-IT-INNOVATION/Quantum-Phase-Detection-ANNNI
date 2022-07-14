""" This module implements the base Hamiltonian class"""
import pennylane as qml
from pennylane import numpy as np
import jax.numpy as jnp

##############

def wall_RZ(active_wires, params, index=0):
    """
    Apply independent RZ rotations to each wire in a Pennylane circuit

    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    # Apply RY to each (active) wire:
    for i, spin in enumerate(active_wires):
        qml.RZ(params[index + i], wires=int(spin) )

    return index + len(active_wires)

def wall_RY(active_wires, params, index=0):
    """
    Apply independent RY rotations to each wire in a Pennylane circuit

    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    # Apply RY to each (active) wire:
    for i, spin in enumerate(active_wires):
        qml.RY(params[index + i], wires=int(spin) )

    return index + len(active_wires)

def wall_RX(active_wires, params, index=0):
    """
    Apply independent RX rotations to each wire in a Pennylane circuit

    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    # Apply RX to each (active) wire:
    for i, spin in enumerate(active_wires):
        qml.RX(params[index + i], wires=int(spin) )

    return index + len(active_wires)

def CRX_neighbour(active_wires, params, index=0):
    """
    Apply independent RX rotations to each wire in a Pennylane circuit

    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    # Apply RX to each (active) wire:
    for i, (spin, spin_next) in enumerate(zip(active_wires, active_wires[1:])):
        qml.CRX(params[index + i], wires=[int(spin), int(spin_next)])

    return index + i + 1

def CRX_nextneighbour(active_wires, params, index=0):
    """
    Apply independent RX rotations to each wire in a Pennylane circuit

    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    # Apply RX to each (active) wire:
    for i, (spin, spin_next) in enumerate(zip(active_wires, active_wires[2:])):
        qml.CRX(params[index + i], wires = [int(spin), int(spin_next)])
        
    return index + i + 1


def wall_CNOT(active_wires):
    """
    Apply CNOTs to every neighbouring (active) qubits

    Parameters
    ----------
    N : int
        Number of qubits
    """
    # Apply entanglement to the neighbouring spins
    for spin, spin_next in zip(active_wires, active_wires[1:]):
        qml.CNOT(wires=[int(spin), int(spin_next)])

def entX_neighbour(active_wires, params, index = 0):
    """ 
    Establish entanglement between qubits using IsingXX gates
    
    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    i = - 1
    # Apply entanglement to the neighbouring spins
    for i, (spin, spin_next) in enumerate(zip(active_wires, active_wires[1:])):
        qml.IsingXX(params[index + i], wires = [int(spin), int(spin_next)])
        
    return index + i + 1

def entX_nextneighbour(active_wires, params, index = 0):
    """ 
    Establish entanglement between qubits using IsingXX gates
    
    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    # Apply entanglement to the neighbouring spins
    i = - 1
    for i, (spin, spin_next) in enumerate(zip(active_wires, active_wires[2:])):
        qml.IsingXX(params[index + i], wires = [int(spin), int(spin_next)])
        
    return index + i + 1


def pooling(active_wires, qmlrot_func, params, index = 0):
    """
    Pooling block for the QCNN

    Parameters
    ----------
    active_wires : np.ndarray
        Array of wires that are not measured during a previous pooling
    params: np.ndarray
        Array of parameters/rotation for the circuit
    qmlrot_func : function
        Pennylane Gate function to apply
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    np.ndarray
        Updated array of active wires (not measured)
    """
    # Pooling:
    isodd = True if len(active_wires) % 2 != 0 else False

    for wire_meas, wire_next in zip(active_wires[0::2], active_wires[1::2]):
        m_0 = qml.measure(int(wire_meas))
        qml.cond(m_0 == 0, qmlrot_func)(params[index], wires=int(wire_next))
        qml.cond(m_0 == 1, qmlrot_func)(params[index + 1], wires=int(wire_next))
        index = index + 2

        # Removing measured wires from active_wires:
        active_wires = np.delete(active_wires, np.where(active_wires == wire_meas))

    # ---- > If the number of wires is odd, the last wires is not pooled
    #        so we apply a gate
    if isodd:
        qmlrot_func(params[index], wires = int(active_wires[-1]) )
        index = index + 1

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
        # Convolution:
        index = wall_RX(active_wires, params, index)
        index = wall_RY(active_wires, params, index)

        # ---- > Establish entanglement: odd connections
        for wire, wire_next in zip(active_wires[0::2], active_wires[1::2]):
            qml.CNOT(wires=[int(wire), int(wire_next)])
            qml.RX(params[index], wires=int(wire))
            index = index + 1

        # ---- > Establish entanglement: even connections
        for wire, wire_next in zip(active_wires[1::2], active_wires[2::2]):
            qml.CNOT(wires=[int(wire), int(wire_next)])
            qml.RX(params[index], wires=int(wire))
            index = index + 1

    qml.RX(params[index], wires=int(active_wires[-1]) )

    return index + 1

def encoder_block(wires, wires_trash, shift = 0):
    """
    Applies CX between a wire and a trash wire for each
    wire/trashwire

    Parameters
    ----------
    N : int
        Number of qubits
    wires : np.ndarray
        Array of the indexes of non-trash qubits
    wires_trash : np.ndarray
        Array of the indexes of trash qubits (np.1dsetdiff(np.arange(N),wires))
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
    for shift in range(len(wires_trash)):
        index = wall_RY(active_wires, params, index)
        encoder_block(wires, wires_trash, shift)
        qml.Barrier()
    index = wall_RY(active_wires, params, index)

    return index
        
def decoder_block(wires_trash, wires_extra, params, index, shift = 0):
    """
    Applies CX between a wire and a trash wire for each
    wire/trashwire

    Parameters
    ----------
    N : int
        Number of qubits
    wires : np.ndarray
        Array of the indexes of non-trash qubits
    wires_trash : np.ndarray
        Array of the indexes of trash qubits (np.1dsetdiff(np.arange(N),wires))
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

def decoder_circuit(wires, wires_trash, wires_extra, params, index = 0):
    index = wall_RY(wires_extra, params, index)
    index = wall_RX(wires_extra, params, index)
    index = wall_RY(wires_extra, params, index)
        
    for shift in range(8):
        index = decoder_block(wires_trash, wires_extra, params, index, shift)
        qml.Barrier()
        
        index = wall_RX(wires_trash, params, index)
        index = wall_RX(wires_extra, params, index)
        
        qml.Barrier()

    for n, m in enumerate(wires_extra):
        qml.SWAP(wires = [int(n),int(m)])
    
    return index

