""" This module implements loss functions and regularizers for VQE, QCNN and Encoder"""
import jax
import jax.numpy as jnp
from jax import jit

# VQE LOSSES
def vqe_fidelities(Y, params, q_circuit):
    """
    LOSS: Compute Fidelity between VQE PSI (output of q_circuit(params)) and TRUE PSI computed by diagonalizing the Hamiltonian
    
    Parameters
    ----------
    Y : np.ndarray
        Array of true wavefunction obtained by diagonalizing the Hamiltonians
    params : np.ndarray
        Array of parameters of the VQE circuits
    q_circuit : fun
        Quantum function of the VQE circuit
        
    Returns
    -------
    float 
        Mean fidelities between VQE PSI and TRUE PSI
    """
    
    # Core function, not vectorized
    def vqe_fidelity(y, p, q_circuit):
        psi_out = q_circuit(p)
        
        return jnp.square(jnp.abs(jnp.conj(psi_out) @  y))
    
    # Vectorize the fidelity function
    v_fidelty = jax.vmap(lambda y, p: vqe_fidelity(y,p,q_circuit), in_axes = (0, 0) )

    return jnp.mean(v_fidelty(Y, params))

def vqe_fidelties_neighbouring(states):
    """
    REGULARIZER: Compute the mean value of the fidelities between each wavefunction and its next one
    
    Parameters
    ----------
    states : np.ndarray
        Array of states (psi)
        
    Returns
    -------
    float
        Mean value of the fidelities between each state and its next one
    """
    def vqe_fidelty(s1, s2):
        return jnp.square(jnp.abs(jnp.conj(s1) @  s2))
    
    v_fidelty = jax.vmap(lambda s1, s2: vqe_fidelty(s1,s2), in_axes = (0, 0) )

    return jnp.mean(v_fidelty(states[:-1], states[1:]))

# QCNN LOSSES
def hinge(X, Y, params, q_circuit):
    """
    LOSS: (Experimental) Compute Hinge loss for a binary classification task
    N.B: MAX is not applied because output is a probability [0,1] that will be mapped
    to [-1,1], hence the 1 - Prediction(X)*Y can be at minimum 0
         
    Parameters
    ----------
    X : np.ndarray
        Array of VQE parameters (input of VQE)
    Y : np.ndarray
        Array of labels
    params : np.ndarray
        Array of parameters of the QCNN circuit
    q_circuit : fun
        Quantum function of the VQE circuit
        
    Returns
    -------
    float
        Mean Hinge Loss <Circuit(X)|Y>
    """
    v_qcnn_prob = jax.vmap(lambda v: q_circuit(v, params))

    predictions = 2*v_qcnn_prob(X) - 1 
    Y_hinge     = 2*Y - 1
    
    hinge_loss = jnp.mean(1 - predictions[:,1]*Y_hinge)
    
    return hinge_loss

def cross_entropy1D(X, Y, params, q_circuit):
    """
    LOSS: Compute Cross Entropy for a binary classification task
    
    Parameters
    ----------
    X : np.ndarray
        Array of VQE parameters (input of VQE)
    Y : np.ndarray
        Array of labels
    params : np.ndarray
        Array of parameters of the QCNN circuit
    q_circuit : fun
        Quantum function of the VQE circuit
        
    Returns
    -------
    float
        Cross entropy <Circuit(X)|Y>
    """
    v_qcnn_prob = jax.vmap(lambda v: q_circuit(v, params))

    predictions = v_qcnn_prob(X)
    logprobs = jnp.log(predictions)

    nll = jnp.take_along_axis(logprobs, jnp.expand_dims(Y, axis=1), axis=1)
    ce = -jnp.mean(nll)

    return ce

def cross_entropy(X, Y, params, q_circuit):
    """
    LOSS: Compute Cross Entropy for a binary classification task
    
    Parameters
    ----------
    X : np.ndarray
        Array of VQE parameters (input of VQE)
    Y : np.ndarray
        Array of labels
    params : np.ndarray
        Array of parameters of the QCNN circuit
    q_circuit : fun
        Quantum function of the VQE circuit
        
    Returns
    -------
    float
        Cross entropy <Circuit(X)|Y>
    """
    v_qcnn_prob = jax.vmap(lambda v: q_circuit(v, params))

    predictions = v_qcnn_prob(X)
    logprobs1 = jnp.log(predictions).flatten()
    logprobs2 = jnp.log(1 - predictions).flatten()
    logprobs1 = jnp.square(jnp.square(logprobs1))
    logprobs2 = jnp.square(jnp.square(logprobs2))
    Y = Y.flatten()
    
    return + jnp.mean( Y*logprobs1 + (1 - Y)* logprobs2)

