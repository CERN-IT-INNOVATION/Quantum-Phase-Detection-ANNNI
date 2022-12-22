""" This module implements the base functions to implement an anomaly detector model"""
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

from matplotlib import pyplot as plt
import matplotlib as mpl

import copy
import tqdm  # Pretty progress bars

from PhaseEstimation import circuits, vqe
from PhaseEstimation import visualization as qplt
from PhaseEstimation import general as qmlgen

from typing import List, Callable
from numbers import Number

##############


def encoder_circuit(N: int, params: List[Number]) -> int:
    """
    Building function for the circuit Encoder(params)

    Parameters
    ----------
    N : int
        Number of qubits
    params: np.ndarray
        Array of parameters/rotation for the circuit

    Returns
    -------
    int
        Number of parameters of the circuit
    """
    active_wires = np.arange(N)

    # Number of wires that will not be measured |phi>
    n_wires = N // 2 + N % 2

    wires = np.concatenate(
        (np.arange(0, n_wires // 2 + n_wires % 2), np.arange(N - n_wires // 2, N))
    )
    wires_trash = np.setdiff1d(active_wires, wires)
    # Visual Separation VQE||Anomaly
    qml.Barrier()
    qml.Barrier()

    index = circuits.encoder_circuit(wires, wires_trash, active_wires, params)

    return index


class encoder:
    def __init__(self, vqe: vqe.vqe, encoder_circuit: Callable):
        """
        Class for the Anomaly Detection algorithm

        Parameters
        ----------
        vqe : class
            VQE class
        encoder_circuit : function
            Function of the Encoder circuit
        """
        self.vqe = vqe
        self.encoder_circuit_fun = lambda enc_p: encoder_circuit(self.vqe.Hs.N, enc_p)
        self.n_params = self.encoder_circuit_fun([0] * 10000)
        self.params = np.array(np.random.rand(self.n_params))
        self.device = vqe.device

        self.vqe_params0 = np.array(vqe.vqe_params0)

        self.n_wires = self.vqe.Hs.N // 2 + self.vqe.Hs.N % 2
        self.n_trash = self.vqe.Hs.N // 2
        self.wires = np.concatenate(
            (
                np.arange(0, self.n_wires // 2 + self.n_wires % 2),
                np.arange(self.vqe.Hs.N - self.n_wires // 2, self.vqe.Hs.N),
            )
        )
        self.wires_trash = np.setdiff1d(np.arange(self.vqe.Hs.N), self.wires)

    def __repr__(self):
        @qml.qnode(self.device, interface="jax")
        def circuit_drawer(self):
            self.encoder_circuit_fun(np.arange(self.n_params))

            return [qml.expval(qml.PauliZ(int(k))) for k in self.wires_trash]

        return qml.draw(circuit_drawer)(self)

    def _vqe_enc_circuit(self, vqe_p: List[Number], qcnn_p: List[Number]):
        self.vqe.circuit(vqe_p)
        self.encoder_circuit_fun(qcnn_p)

    def train(
        self, lr: Number, n_epochs: int, train_index: List[int], circuit: bool = False
    ):
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
        """
        if circuit:
            # Display the circuit
            print("+--- CIRCUIT ---+")
            print(self)

        # Get the index of the training VQE states
        X_train = jnp.array(self.vqe_params0[train_index])

        @qml.qnode(self.device, interface="jax")
        def q_encoder_circuit(vqe_params, params):
            self._vqe_enc_circuit(vqe_params, params)

            return [qml.expval(qml.PauliZ(int(k))) for k in self.wires_trash]

        v_q_encoder_circuit = jax.vmap(
            lambda p, x: q_encoder_circuit(x, p), in_axes=(None, 0)
        )

        def compress(params, vqe_params):
            return jnp.sum(1 - v_q_encoder_circuit(params, vqe_params)) / (
                2 * len(vqe_params)
            )

        jd_compress = jax.jit(jax.grad(lambda p: compress(p, X_train)))
        j_compress = jax.jit(lambda p: compress(p, X_train))

        def update(params, opt_state):
            grads = jd_compress(params)
            opt_state = opt_update(0, grads, opt_state)

            return get_params(opt_state), opt_state

        params = copy.copy(self.params)

        # Defining an optimizer in Jax
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(params)

        progress = tqdm.tqdm(range(n_epochs), position=0, leave=True)
        for epoch in range(n_epochs):
            params, opt_state = update(params, opt_state)

            if (epoch + 1) % 100 == 0:
                loss = j_compress(params)
                progress.set_description("Cost: {0}".format(loss))
            progress.update(1)

        self.params = params

    def show_compression(self, trainingpoint, label = False, plot3d = False):
        qplt.ENC_show_compression_ANNNI(self, trainingpoint=trainingpoint, label=label, plot3d=plot3d)

def enc_classification_ANNNI(
    vqeclass: vqe.vqe, lr: Number, epochs: int
) -> List[Number]:
    """
    Train 3 encoder on the corners: 
    > K = 0,  L = 2 (Paramagnetic)
    > K = 0,  L = 0 (Ferromagnetic)
    > K = -1, L = 0 (Antiphase)
    The other states will be classified taking the lowest error among each encoder
    
    Parameters
    ----------
    vqeclass : class
        VQE class
    lr : float
        Learning rate for each training
    epochs : int
        Number of epochs for each training
        
    Returns
    -------
    np.ndarray
        Array of labels
    """
    # indexes of the 3 corner points
    side = vqeclass.Hs.side
    phase1 = 0
    phase2 = side - 1
    phase3 = int(vqeclass.Hs.n_states - side)

    encclass = encoder(vqeclass, encoder_circuit)

    X = jnp.array(encclass.vqe_params0)

    @qml.qnode(encclass.device, interface="jax")
    def encoder_circuit_class(vqe_params, params):
        encclass._vqe_enc_circuit(vqe_params, params)

        return [qml.expval(qml.PauliZ(int(k))) for k in encclass.wires_trash]

    encoding_scores = []

    for phase in [phase1, phase2, phase3]:
        encclass = encoder(vqeclass, encoder_circuit)
        encclass.train(lr, epochs, np.array([phase]), circuit=False)
        v_encoder_circuit = jax.vmap(
            lambda x: encoder_circuit_class(x, encclass.params)
        )
        exps = (1 - np.sum(v_encoder_circuit(X), axis=1) / 4) / 2
        exps = np.rot90(np.reshape(exps, (side, side)))

        encoding_scores.append(exps)

    qplt.getlines(qmlgen.paraanti, [0.5, 1 - 1e-5], side, "white", res=100)
    qplt.getlines(qmlgen.paraferro, [1e-5, 0.5], side, "white", res=100)

    phases = mpl.colors.ListedColormap(["navy", "crimson", "limegreen", "limegreen"])
    norm = mpl.colors.BoundaryNorm(np.arange(0, 4), phases.N)
    plt.imshow(np.argmin(np.array(encoding_scores), axis=(0)), cmap=phases, norm=norm)

    plt.ylabel(r"$h$", fontsize=24)
    plt.xlabel(r"$\kappa$", fontsize=24)

    plt.xticks(
        ticks=np.linspace(0, side - 1, 5).astype(int),
        labels=[np.round(k * 1 / 4, 2) for k in range(0, 5)],
        fontsize=18,
    )
    plt.yticks(
        ticks=np.linspace(0, side - 1, 5).astype(int),
        labels=[np.round(k * 2 / 4, 2) for k in range(4, -1, -1)],
        fontsize=18,
    )

    return np.argmin(np.array(encoding_scores), axis=(0))
