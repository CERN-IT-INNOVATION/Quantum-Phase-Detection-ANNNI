""" This module implements the base functions for the Quantum Anomaly Detection"""
import pennylane as qml
from pennylane import numpy as np
from jax import jit, vmap, value_and_grad
from jax import numpy as jnp
import optax

from PhaseEstimation import annni, circuits

from typing import Callable
import tqdm
import time

import matplotlib.pyplot as plt

class Ad:
    def __init__(self, n_qubit: int, side : int, ansatz: Callable = circuits.anomaly, vqeclass = None):
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

        self.n_p, self.p_trashwire = self.ansatz(n_qubit, np.arange(10000))
        self.q_circuit   = qml.QNode(self.circuit, device=self.device) 
        self.jq_circuit  = jit(self.q_circuit)
        self.vjq_circuit = vmap(self.jq_circuit, in_axes=(None, 0))

        self.p_p = np.random.normal(loc=0, scale=1, size=self.n_p)

        self.p_h = np.linspace(0,2,self.side)
        self.p_k = np.linspace(0,1,self.side)
        
        p_state = []
        self.index_map = []
        
        progress = tqdm.tqdm(range(self.side*self.side))
        i = 0
        for k in self.p_k:
            for h in self.p_h:
                self.index_map.append((k, h))
                if self.vqeclass is None:
                    H = annni.Annni(self.n_qubit, k, h)
                    p_state.append(H.psi)
                else:
                    p_state.append(self.vqeclass.dict_p_p[(float(k), float(h))])

                progress.set_description(f"Inizialization: k: {k:.2f} | h: {h:.2f}")
                progress.update(1)
                i += 1
                
        self.p_state = jnp.array(p_state)

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
        return [qml.expval(qml.PauliZ(int(k))) for k in self.p_trashwire]

    def train(self, n_epoch: int, lr: float, reset: bool = False, h : float = 0, k : float = 0):
        # Find the closest value in p_k and p_h
        k_closest = self.p_k[np.argmin(np.abs(self.p_k - k))]
        h_closest = self.p_h[np.argmin(np.abs(self.p_h - h))]
        
        self.train_cord = (float(k_closest), float(h_closest))
        
        state = self.p_state[self.index_map.index((k_closest, h_closest))]

        def _loss(p_p):
            # Output expectation values of the qubits
            score = 1 - jnp.array(self.jq_circuit(p_p, state))
            loss_value = jnp.sum(score)

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
                f"COST: {loss_value:.4f}"
            )

        # Time ending training
        t_train_stop = time.time()

        # At the end of the training, set the attribute params to the
        # trained parameters
        self.p_p = p_p

        self.training_time = t_train_stop - t_train_start

    def show(self, mpl = False):
        if mpl:
            qml.draw_mpl(self.q_circuit)(np.arange(self.n_p), np.array(self.p_state[0]))
        else:
            print(qml.draw(self.q_circuit)(np.arange(self.n_p), np.array(self.p_state[0])))
            
    def __repr__(self):
        repr_str  = "Anomaly Detection Class:"
        repr_str += f"\n  N        : {self.n_qubit}"
        repr_str += f"\n  side     : {self.side}"
        repr_str += f"\n  n_params : {self.n_p}"
        
        return repr_str
    
    def predict(self):
        """
        Output the compression score on the full phase space
        """
        if hasattr(self, "train_cord"):
            p_compression = jnp.sum(1 - jnp.array(self.vjq_circuit(self.p_p, self.p_state)), axis=0)

            # Plot the classification results
            plt.figure(figsize=(4.5,4))

            plt.imshow(np.flip(np.rot90(p_compression.reshape(-1, self.side), k=-1),axis=1), aspect="auto", origin="lower", extent=[0, 1, 0, 2])
            plt.colorbar()
            plt.scatter([self.train_cord[0] + .3/len(self.p_k)], [self.train_cord[1] + .5/len(self.p_h)], color='red', marker="x", label="Training Point")
            annni.set_layout('Compression')

            plt.show()
        else:
            raise RuntimeError("No training point found, model has to be trained through cls.train function")

