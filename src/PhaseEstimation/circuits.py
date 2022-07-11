""" This module implements the base Hamiltonian class"""
import pennylane as qml
from pennylane import numpy as np
import jax.numpy as jnp

##############

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

