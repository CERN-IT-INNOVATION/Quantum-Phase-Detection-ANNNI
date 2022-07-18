""" This module implements the base functions to implement an autoencoder"""
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.example_libraries import optimizers

from matplotlib import pyplot as plt

import copy
import tqdm  # Pretty progress bars

import warnings

warnings.filterwarnings(
    "ignore",
    message="For Hamiltonians, the eigenvalues will be computed numerically. This may be computationally intensive for a large number of wires.Consider using a sparse representation of the Hamiltonian with qml.SparseHamiltonian.",
)

import sys, os
sys.path.insert(0, '../../')
import PhaseEstimation.circuits as circuits
import PhaseEstimation.losses as losses
import PhaseEstimation.vqe as vqe

##############

def autoencoder_circuit(N, vqe_circuit, vqe_params, params, decode = True):
    """
    Building function for the circuit:
          VQE(params_vqe) + Anomaly(params)

    Parameters
    ----------
    N : int
        Number of qubits
    vqe_circuit : function
        Function of the VQE Circuit
    vqe_params : np.ndarray
        Array of VQE parameters (states)
    params: np.ndarray
        Array of parameters/rotation for the circuit

    Returns
    -------
    np.ndarray
        Index of the trash qubits
    int
        Number of parameters of the circuit
    """
    active_wires = np.arange(N)
    
    # Number of wires that will not be measured |phi>
    n_wires = N - 3
    # Number of wires that will be measured |0>^k
    n_trash = 3

    wires = np.arange(N-2)
    wires_trash = np.arange(N-2,N)
    wires_extra = np.arange(N, 2*N - 2)

    vqe_circuit(N, vqe_params)

    # Visual Separation VQE||Anomaly
    qml.Barrier()
    qml.Barrier()
    
    index = circuits.encoder_circuit(wires, wires_trash, active_wires, params)
    
    qml.Barrier()
    qml.Barrier()
    
    if decode:
        index = circuits.decoder_circuit(wires, wires_trash, wires_extra, params, index)

        return index
    return index


class autoencoder:
    def __init__(self, vqe, encoder_circuit):
        self.N = vqe.N
        self.vqe = vqe
        self.n_states = vqe.n_states
        self.circuit = lambda vqe_p, enc_p, decode: encoder_circuit(
            self.N, vqe.circuit_fun, vqe_p, enc_p, decode
        )
        self.n_params = self.circuit([0] * 10000, [0] * 10000, decode = True)
        self.params = np.array([np.pi / 4] * self.n_params)
        self.device = qml.device("default.qubit.jax", wires=2*(self.N - 1), shots=None) 

        self.vqe_params = np.array(vqe.vqe_params)
        self.train_index = []
        self.circuit_fun = encoder_circuit
        self.n_wires = self.N - 3
        self.n_trash = 3
        self.wires = np.arange(self.N-3)
        self.wires_trash = np.arange(self.N-3,self.N)

    def show_circuit(self):
        """
        Prints the current circuit defined by self.circuit
        """

        @qml.qnode(self.device, interface="jax")
        def enc_state(self):
            self.circuit([0] * 1000, np.arange(self.n_params), decode = True)

            return qml.state()

        drawer = qml.draw(enc_state)
        print(drawer(self))
        
    def get_fidelties_autoencoder(self, params):
        @qml.qnode(self.device, interface="jax")
        def psi_out(x):
            self.circuit(x, params, decode = True)

            return qml.state()
        
        @qml.qnode(self.vqe.device, interface="jax")
        def psi_in(x):
            self.vqe.circuit(x)

            return qml.state()
        
        def fidelty(params, x):
            state_autoencoder = psi_out(x)[:2**self.vqe.N]
            state_input = psi_in(x)
            fidelty = jnp.square(jnp.abs( jnp.conj(state_autoencoder) @  state_input))

            return fidelty
        
        jv_fidelty = jax.jit(jax.vmap( fidelty, in_axes = (None, 0) ))
            
        return jnp.mean(jv_fidelty(params, self.vqe.vqe_params))

    def train(self, lr, n_epochs, train_index, circuit=False, plot=False):
        """
        Training function for the Anomaly Detector.

        Parameters
        ----------
        lr : float
            Learning rate to be multiplied in the circuit-gradient output
        n_epochs : int
            Total number of epochs for each learning
        train_index : np.ndarray
            Index of training points
        circuit : bool
            if True -> Prints the circuit
            if False -> It does not display the circuit
        plot : bool
            if True -> Display the loss curve
        """
        if circuit:
            # Display the circuit
            print("+--- CIRCUIT ---+")
            self.show_circuit()

        X_train = jnp.array(self.vqe_params[train_index])
        
        @qml.qnode(self.vqe.device, interface="jax")
        def vqe_state(vqe_params):
            self.vqe.circuit(vqe_params)

            return qml.state()
        
        jv_vqe_state = jax.jit(jax.vmap(vqe_state))
        Y_train = jv_vqe_state(X_train)
        
        @qml.qnode(self.device, interface="jax")
        def q_autoencoder_circuit(vqe_params, params):
            self.circuit(vqe_params, params, decode = True)

            # return <psi|H|psi>
            return qml.state()

        v_q_autoencoder_circuit = jax.vmap(
            lambda p, x: q_autoencoder_circuit(x, p), in_axes=(None, 0)
        )

        def reconstruct(X, Y, params):
            #return jnp.mean( jnp.square( jnp.abs( v_q_autoencoder_circuit(params, X)[:,:2**self.vqe.N] - Y) ) )
            return 1 - self.get_fidelties_autoencoder(params)
        
        jd_reconstruct = jax.jit(jax.grad(lambda p: reconstruct(X_train, Y_train, p)))
        j_reconstruct = jax.jit(lambda p: reconstruct(X_train, Y_train, p))
        
        def update(params, opt_state):
            grads = jd_reconstruct(params)
            opt_state = opt_update(0, grads, opt_state)
            
            return get_params(opt_state), opt_state
        
        params = copy.copy(self.params)
        
        # Defining an optimizer in Jax
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(params)

        loss = []
        progress = tqdm.tqdm(range(n_epochs), position=0, leave=True)
        for epoch in range(n_epochs):
            params, opt_state = update(params, opt_state)

            if (epoch + 1) % 100 == 0:
                loss.append(j_reconstruct(params))
                progress.set_description("Cost: {0} | Mean F. {1}".format(loss[-1], self.get_fidelties_autoencoder(params)))
            progress.update(1)

        self.params = params
        self.train_index = train_index

        if plot:
            plt.title("Loss of the encoder")
            plt.plot(np.arange(len(loss)) * 10, loss)
        
    def show_latent_space(self):
        @qml.qnode(self.device, interface="jax")
        def q_latent_probs(vqe_p, auto_p):
            self.circuit(vqe_p, auto_p, decode = False)

            # return <psi|H|psi>
            return [qml.probs(wires = int(tr)) for tr in self.wires_trash]
        
        jv_q_latent_probs = jax.jit(jax.vmap(lambda x, p : q_latent_probs(x, p), in_axes=(0,None)))
        
        latent_vectors = jv_q_latent_probs(self.vqe.vqe_states, self.params)
        
        x = latent_vectors[:,0,0]
        y = latent_vectors[:,1,0]
        
        c = []
        for label in self.vqe.Hs.labels:
            if label == 0:
                c.append('green')
            else:
                c.append('blue')
        
        plt.scatter(x,y, color = c)
        
    def latent_classify(self, lr, n_epochs, train_index, loss_fn, plot = True):
        X_train = jnp.array(self.vqe_states[train_index])
        Y_train = jnp.array(self.vqe.Hs.labels)[train_index]
        
        test_index = np.setdiff1d(np.arange(len(self.vqe_states)), train_index)
        X_test = jnp.array(self.vqe_states[test_index])
        Y_test = jnp.array(self.vqe.Hs.labels)[test_index]
        
        def latent_qcnn_circuit(vqe_p, qcnn_p):    
            self.circuit(vqe_p, self.params, decode = False)

            index = 0
            qml.CNOT(wires = [int(self.wires_trash[-1]), int(self.wires_trash[-2]) ])
            index = circuits.wall_RY([self.wires_trash[-1],self.wires_trash[-2]], qcnn_p, index)
            index = circuits.wall_RX([self.wires_trash[-1],self.wires_trash[-2]], qcnn_p, index)
            index = circuits.wall_RY([self.wires_trash[-1],self.wires_trash[-2]], qcnn_p, index)

            m_0 = qml.measure(int(self.wires_trash[-2]))
            qml.cond(m_0 == 0, qml.RY)(qcnn_p[index], wires=int(self.wires_trash[-1]))
            index += 1
            qml.cond(m_0 == 1, qml.RY)(qcnn_p[index], wires=int(self.wires_trash[-1]))
            index += 1

            qml.RY(qcnn_p[index], wires = int(self.wires_trash[-1]) )

            return index + 1

        qcnn_n_params = latent_qcnn_circuit([0]*1000, [0]*1000)

        @qml.qnode(self.device, interface="jax")
        def q_latent_qcnn_circuit(vqe_p, qcnn_p):
            latent_qcnn_circuit(vqe_p, qcnn_p)

            # return <psi|H|psi>
            return qml.probs(wires = int(self.wires_trash[-1]))

        # Gradient of the Loss function
        jd_loss_fn = jax.jit(
            jax.grad(lambda p: loss_fn(X_train, Y_train, p, q_latent_qcnn_circuit))
        )

        def update(params, opt_state):
            grads = jd_loss_fn(params)
            opt_state = opt_update(0, grads, opt_state)

            return get_params(opt_state), opt_state

        # Compute Loss of whole sets
        train_loss_fn = jax.jit(
            lambda p: loss_fn(X_train, Y_train, p, q_latent_qcnn_circuit)
        )
        test_loss_fn = jax.jit(
            lambda p: loss_fn(X_test, Y_test, p, q_latent_qcnn_circuit)
        )

        params = np.random.rand(qcnn_n_params)

        progress = tqdm.tqdm(range(n_epochs), position=0, leave=True)

        # Defining an optimizer in Jax
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(params)

        loss_history = []
        loss_history_test = []
        for epoch in range(n_epochs):
            params, opt_state = update(params, opt_state)

            if epoch % 100 == 0:
                loss_history.append(train_loss_fn(params))
                if len(Y_test) > 0:
                    loss_history_test.append(test_loss_fn(params))
            progress.update(1)
            progress.set_description("Cost: {0}".format(loss_history[-1]))

        self.loss_train_qcnn = loss_history
        self.loss_test_qcnn = loss_history_test
        self.params_qcnn = params
        self.train_index = train_index

        if plot:
            plt.figure(figsize=(15, 5))
            plt.plot(
                np.arange(len(loss_history)) * 100,
                np.asarray(loss_history),
                label="Training Loss",
            )
            if len(X_test) > 0:
                plt.plot(
                    np.arange(len(loss_history_test)) * 100,
                    np.asarray(loss_history_test),
                    label="Test Loss",
                )
            plt.axhline(y=0, color="r", linestyle="--")
            plt.title("Loss history")
            plt.ylabel("Average Cross entropy")
            plt.xlabel("Epoch")
            plt.grid(True)
            plt.legend()

            test_index = np.setdiff1d(np.arange(len(self.vqe_states)), train_index)

            predictions_train = []
            predictions_test = []

            colors_train = []
            colors_test = []

            vcircuit = jax.vmap(lambda v: q_latent_qcnn_circuit(v, self.params_qcnn), in_axes=(0))
            predictions = vcircuit(self.vqe_states)[:, 1]

            for i, prediction in enumerate(predictions):
                # if data in training set
                if i in train_index:
                    predictions_train.append(prediction)
                    if np.round(prediction) == 0:
                        colors_train.append("green") if self.vqe.Hs.labels[
                            i
                        ] == 0 else colors_train.append("red")
                    else:
                        colors_train.append("red") if self.vqe.Hs.labels[
                            i
                        ] == 0 else colors_train.append("green")
                else:
                    predictions_test.append(prediction)
                    if np.round(prediction) == 0:
                        colors_test.append("green") if self.vqe.Hs.labels[
                            i
                        ] == 0 else colors_test.append("red")
                    else:
                        colors_test.append("red") if self.vqe.Hs.labels[
                            i
                        ] == 0 else colors_test.append("green")

            fig, ax = plt.subplots(2, 1, figsize=(16, 10))

            ax[0].set_xlim(-0.1, 2.1)
            ax[0].set_ylim(0, 1)
            ax[0].grid(True)
            ax[0].axhline(y=0.5, color="gray", linestyle="--")
            ax[0].axvline(x=1, color="gray", linestyle="--")
            ax[0].text(0.375, 0.68, "I", fontsize=24, fontfamily="serif")
            ax[0].text(1.6, 0.68, "II", fontsize=24, fontfamily="serif")
            ax[0].set_xlabel("Transverse field")
            ax[0].set_ylabel("Prediction of label II")
            ax[0].set_title("Predictions of labels; J = 1")
            ax[0].scatter(
                2 * np.sort(train_index) / len(self.vqe.vqe_states),
                predictions_train,
                c="royalblue",
                label="Training samples",
            )
            ax[0].scatter(
                2 * np.sort(test_index) / len(self.vqe.vqe_states),
                predictions_test,
                c="orange",
                label="Test samples",
            )
            ax[0].legend()

            ax[1].set_xlim(-0.1, 2.1)
            ax[1].set_ylim(0, 1)
            ax[1].grid(True)
            ax[1].axhline(y=0.5, color="gray", linestyle="--")
            ax[1].axvline(x=1, color="gray", linestyle="--")
            ax[1].text(0.375, 0.68, "I", fontsize=24, fontfamily="serif")
            ax[1].text(1.6, 0.68, "II", fontsize=24, fontfamily="serif")
            ax[1].set_xlabel("Transverse field")
            ax[1].set_ylabel("Prediction of label II")
            ax[1].set_title("Predictions of labels; J = 1")
            ax[1].scatter(
                2 * np.sort(train_index) / len(self.vqe.vqe_states),
                predictions_train,
                c=colors_train,
            )
            ax[1].scatter(
                2 * np.sort(test_index) / len(self.vqe.vqe_states),
                predictions_test,
                c=colors_test,
            )