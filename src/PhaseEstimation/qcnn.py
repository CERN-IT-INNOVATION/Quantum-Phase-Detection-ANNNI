"""This module implements the base functions to implement a Quantum Convolutional Neural Network (QCNN) for the (ANNNI) Ising Model. """
import pennylane as qml
from pennylane import numpy as np
from jax import jit, vmap, value_and_grad
from jax import numpy as jnp
from jax import config
import optax

from PhaseEstimation import annni, circuits

from typing import Callable
import tqdm
import time

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Training raises UserWarning: 
# Explicitly requested dtype <class 'jax.numpy.complex128'> 
# requested in astype is not available, and will be truncated 
# to dtype complex64.
# TODO: Is complex64 sufficient? Does it bring more speed?
config.update("jax_enable_x64", True)

def cross_entropy(pred : np.ndarray, Y : np.ndarray, T : float = 0.5):
    """
    Computes the categorical cross-entropy loss with temperature scaling
    to encourage more extreme predictions.

    Parameters
    ----------
    pred : np.ndarray
        Array of the predicted probabilities.
    Y : np.ndarray
        one-hot encoded true labels.
    T : float
        temperature parameter (<1 makes the model more confident).

    Returns
    -------
    float
        Scalar value, the mean categorical cross-entropy loss.
    """
    epsilon = 1e-9  # Small value for numerical stability
    pred = jnp.clip(pred, epsilon, 1 - epsilon)  # Prevent log(0)
    
    # Apply sharpening (raise probabilities to the power of 1/T)
    pred_sharpened = pred ** (1 / T)
    pred_sharpened /= jnp.sum(pred_sharpened, axis=1, keepdims=True)  # Re-normalize
    
    loss = -jnp.sum(Y * jnp.log(pred_sharpened), axis=1)  # Compute cross-entropy
    return jnp.mean(loss)  # Return mean loss over batch

class Qcnn:
    def __init__(self, n_qubit: int, side : int, ansatz: Callable = circuits.qcnn, vqeclass = None):
        """
        Initialize the Anomaly Detection with given parameters.

        Parameters
        ----------
        n_spin : int
            Number of spins of the system
        side : int
            Discretization of the phase space
        ansatz : Callable
            Pennylane circuit ansatz
        vqeclass : PhaseEstimation.vqe.Vqe 
            Trained VQE model for the inputs. If None, inputs will be obtained by 
            diagonalizing the corresponding hamiltonian
        """
        self.vqeclass = vqeclass
        self.n_qubit = n_qubit if self.vqeclass is None else self.vqeclass.n_qubit
        self.side = side if self.vqeclass is None else self.vqeclass.side
        self.ansatz = ansatz

        self.device = qml.device("default.qubit", wires=self.n_qubit, shots=None)

        self.n_p, self.p_outputwire = self.ansatz(n_qubit, np.arange(10000))
        self.q_circuit   = qml.QNode(self.circuit, device=self.device) 
        self.jq_circuit  = jit(self.q_circuit)
        self.vjq_circuit = vmap(self.jq_circuit, in_axes=(None, 0))

        self.p_p = np.random.normal(loc=0, scale=1, size=self.n_p)

        self.p_h = np.linspace(0,2,self.side)
        self.p_k = np.linspace(0,1,self.side)
        
        p_state = []
        p_Y = []
        analytical_mask = []
        self.index_map = []
        
        progress = tqdm.tqdm(range(self.side*self.side))
        i = 0
        for k in self.p_k:
            for h in self.p_h:
                self.index_map.append((k, h))
                if self.vqeclass is None:
                    H = annni.Annni(self.n_qubit, k, h)
                    p_state.append(H.psi)
                    p_Y.append(jnp.eye(4)[H.phase])
                else:
                    p_state.append(self.vqeclass.dict_p_p[(float(k), float(h))])
                    p_Y.append(jnp.eye(4)[self.vqeclass.dict_Y[(float(k), float(h))]])
                    
                if k == 0 or h == 0:
                    analytical_mask.append(i)

                progress.set_description(f"Inizialization: k: {k:.2f} | h: {h:.2f}")
                progress.update(1)
                i += 1
                
        self.p_state = jnp.array(p_state)
        self.p_Y = jnp.array(p_Y)
        self.analytical_mask = jnp.array(analytical_mask).astype(int)

    def ansatz_combined(self, p_p, state):
        if self.vqeclass is None:
            qml.StatePrep(state, wires=range(self.n_qubit), normalize = True)
        else:
            self.vqeclass.ansatz(self.n_qubit, state, **self.vqeclass.kwargs)

        # Visual Separation VQE||QCNN
        qml.Barrier()
        qml.Barrier()

        self.ansatz(self.n_qubit, p_p)

    def circuit(self, p_p, state):
        self.ansatz_combined(p_p=p_p, state=state)
        return qml.probs([int(k) for k in self.p_outputwire])

    def train(self, n_epoch: int, lr: float, reset: bool = False, T : float = .25):
        p_X = self.p_state[self.analytical_mask]
        p_Y = self.p_Y[self.analytical_mask]

        def _loss(p_p):
            # Output expectation values of the qubits
            p_pred = self.vjq_circuit(p_p, p_X)
            loss_value = cross_entropy(p_pred, p_Y, T=T)

            return loss_value

        def _update(
            optimizer,
            state,
            p_p,
        ):
            ce_loss, grads = value_and_grad(_loss)(p_p)
            updates, optimizer_state = optimizer.update(grads, state)
            p_p = optax.apply_updates(p_p, updates)

            return p_p, optimizer_state, ce_loss

        # Redraw random parameters if True
        if reset:
            self.p_p = np.random.normal(loc=0, scale=1, size=self.n_p)
            
        p_p = self.p_p

        # Set the optimizer
        optimizer = optax.adam(learning_rate=lr)
        optimizer_state = optimizer.init(p_p)

        progress = tqdm.tqdm(range(1, n_epoch + 1))

        # Time start training
        t_train_start = time.time()
        
        for epoch in progress:
            p_p, optimizer_state, loss_value = _update(
                optimizer,
                optimizer_state,
                p_p,
            )

            progress.set_description(
                f"LOSS: {loss_value:.4f}"
            )

        # Time ending training
        t_train_stop = time.time()

        # At the end of the training, set the attribute params to the
        # trained parameters
        self.p_p = p_p

        self.training_time = t_train_stop - t_train_start

    def show(self, mpl=False):
        if mpl:
            qml.draw_mpl(self.q_circuit)(np.arange(self.n_p), np.array(self.p_state[0]))
        else:
            print(qml.draw(self.q_circuit)(np.arange(self.n_p), np.array(self.p_state[0])))
            
    def __repr__(self):
        repr_str  = "QCNN Class:"
        repr_str += f"\n  N        : {self.n_qubit}"
        repr_str += f"\n  side     : {self.side}"
        repr_str += f"\n  n_params : {self.n_p}"
        
        return repr_str
    
    def predict(self):
        """
        Ouput the predicted phase of every input state in the phase space 
        """
        p_pred = self.vjq_circuit(self.p_p, self.p_state)
        accuracy = jnp.mean(jnp.argmax(p_pred, axis=1) == jnp.argmax(self.p_Y, axis=1))
        print(f"Accuracy: {float(accuracy) * 100:.2f}%")

        # Ensure reshaping is valid
        img = np.argmax(p_pred, axis=1).reshape(-1, self.side)

        # Plot the classification results
        plt.figure(figsize=(4,4))

        colors = ['#4d94d7', '#d74d94', '#68d16c', '#ffd34d']

        # Create a custom colormap
        cmap = ListedColormap(colors)
        plt.imshow(np.flip(np.rot90(img, k=-1),axis=1), cmap = cmap, aspect="auto", origin="lower", extent=[0, 1, 0, 2])
        annni.set_layout('Classification')
        plt.show()

        for i, label in enumerate(['Paramagnetic', 'Ferromagnetic', 'Antiphase', 'FloatingPhase']):
            img = p_pred[:, i].reshape(-1, self.side)
            plt.figure(figsize=(4.5,4))
            plt.imshow(np.flip(np.rot90(img, k=-1),axis=1), cmap = "viridis", aspect="auto", origin="lower", extent=[0, 1, 0, 2])
            annni.set_layout(label)
            plt.colorbar()
            plt.show()

        return p_pred