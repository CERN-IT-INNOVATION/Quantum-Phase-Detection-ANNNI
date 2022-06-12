### IMPORTS ###
# Quantum libraries:
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# Plotting
from matplotlib import pyplot as plt

# Other
import copy
from tqdm.notebook import tqdm # Pretty progress bars
from IPython.display import Markdown, display # Better prints
import joblib # Writing and loading
from noisyopt import minimizeSPSA

import multiprocessing
##############

import vqe_functions as vqe

#   _    
#  / |   
#  | |   
#  | |_  
#  |_(_) 

def num_params_qcnn(N):
    '''
    N = number of wires (spins)
    To evaluate the number of parameters needed for the qcnn
    a recursive function is needed:
    '''
    n_params = 0
    # While the number of wires is more than 1s
    while(N > 1):
        # Convolution
        n_params += 3*N
        # Pooling 
        n_params += 2*(N//2) + N%2
        # Reduce number of wires due to pooling
        N = N // 2 + N % 2
    
    # Last RY gate
    n_params += 1
    
    return n_params

#  ____     
# |___ \    
#   __) |   
#  / __/ _  
# |_____(_) Circuit functions

def qcnn_convolution(active_wires, params, N, p_index, conv_noise = 0):
    '''
    Convolution block for the QCNN
    
    RX--RY--o--RX---------
            | 
    RX--RY--x------o--RX--
                   |
    RX--RY--o--RX--x------
            | 
    RX--RY--x------o--RX--
                   |
    RX--RY--o--RX--x------
            | 
    RX--RY--x---------RX--
    
    '''
    
    # Check if the current number of wires is odd
    # it will be needed later.
    isodd = True if len(active_wires) % 2 != 0  else False
    
    noise = True
    if conv_noise == 0: noise = False # Remove BitFlip and PhaseFlip if we are not using default.mixed
    
    # Convolution:
    for wire in active_wires:
        qml.RX(params[p_index],   wires = int(wire) )
        qml.RY(params[p_index+1], wires = int(wire) )
        p_index = p_index + 2
        
        if noise: qml.PhaseFlip(conv_noise, wires = int(wire) ); qml.BitFlip(conv_noise, wires = int(wire) )
        
    # ---- > Establish entanglement: odd connections
    for wire, wire_next in zip(active_wires[0::2], active_wires[1::2]):
        qml.CNOT(wires = [int(wire), int(wire_next)])
        qml.RX(params[p_index], wires = int(wire) )
        p_index = p_index + 1
        
        if noise: qml.PhaseFlip(conv_noise, wires = int(wire) ); qml.BitFlip(conv_noise, wires = int(wire) )
        
    # ---- > Establish entanglement: even connections
    for wire, wire_next in zip(active_wires[1::2], active_wires[2::2]):
        qml.CNOT(wires = [int(wire), int(wire_next)])
        qml.RX(params[p_index], wires = int(wire) )
        p_index = p_index + 1
        
        if noise: qml.PhaseFlip(conv_noise, wires = int(wire) ); qml.BitFlip(conv_noise, wires = int(wire) )
        
    qml.RX(params[p_index], wires = N-1)
    p_index = p_index + 1
    
    if noise: qml.PhaseFlip(conv_noise, wires = int(N - 1) ); qml.BitFlip(conv_noise, wires = int(N - 1) )

    return p_index
        
def qcnn_pooling(active_wires, params, N, p_index, pool_noise = 0):
    '''
    Pooling block for the QCNN
    
    --MEAS--(=0)--(=1)
             |     |
    ---------RY----RZ----
    
    '''
    # Pooling:
    isodd = True if len(active_wires) % 2 != 0  else False
    
    noise = True
    if pool_noise == 0: noise = False # Remove BitFlip and PhaseFlip if we are not using default.mixed
    
    for wire_meas, wire_next in zip(active_wires[0::2], active_wires[1::2]):
        m_0 = qml.measure(int(wire_meas) )
        qml.cond(m_0 ==0, qml.RY)(params[p_index], wires=int(wire_next) )
        qml.cond(m_0 ==1, qml.RZ)(params[p_index+1], wires=int(wire_next) )
        p_index = p_index + 1
        
        if noise: qml.PhaseFlip(pool_noise, wires = int(wire_next) ); qml.BitFlip(pool_noise, wires = int(wire_next) )
        
        # Removing measured wires from active_wires:
        active_wires = np.delete(active_wires, np.where(active_wires == wire_meas) ) 
    # ---- > If the number of wires is odd, the last wires is not pooled
    #        so we apply a Z gate
    if isodd:
        qml.RZ(params[p_index], wires = N-1)
        p_index = p_index + 1
        
        if noise: qml.PhaseFlip(pool_noise, wires = N - 1 ); qml.BitFlip(pool_noise, wires = N - 1 )
        
    return p_index, active_wires

#  _____   
# |___ /   
#   |_ \   
#  ___) |  
# |____(_) Learning functions