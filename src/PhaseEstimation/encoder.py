""" This module implements the base functions to implement an anomaly detector"""
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.example_libraries import optimizers

from matplotlib import pyplot as plt
import matplotlib as mpl

import copy
import tqdm  # Pretty progress bars

import warnings

warnings.filterwarnings(
    "ignore",
    message="For Hamiltonians, the eigenvalues will be computed numerically. This may be computationally intensive for a large number of wires.Consider using a sparse representation of the Hamiltonian with qml.SparseHamiltonian.",
)

import sys
sys.path.insert(0, '../../')
import PhaseEstimation.circuits as circuits

##############

def encoder_circuit(N, params):
    """
    Building function for the circuit:
          VQE(params_vqe) + Encoder(params)

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
    n_wires = N // 2 + N % 2
    # Number of wires that will be measured |0>^k
    n_trash = N // 2

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
    def __init__(self, vqe, encoder_circuit):
        """
        Class for the Anomaly Detection algorithm

        Parameters
        ----------
        vqe : class
            VQE class
        encoder_circuit :
            Function of the Encoder circuit
        """
        self.N = vqe.N
        self.vqe = vqe
        self.n_states = vqe.n_states
        self.encoder_circuit_fun = lambda enc_p: encoder_circuit(
            self.N, enc_p
        )
        self.n_params = self.encoder_circuit_fun([0] * 10000)
        self.params = np.array(np.random.rand(self.n_params) )
        self.device = vqe.device

        self.vqe_params0 = np.array(vqe.vqe_params0)
        self.train_index = []
        self.n_wires = self.N // 2 + self.N % 2
        self.n_trash = self.N // 2
        self.wires = np.concatenate(
            (
                np.arange(0, self.n_wires // 2 + self.n_wires % 2),
                np.arange(self.N - self.n_wires // 2, self.N),
            )
        )
        self.wires_trash = np.setdiff1d(np.arange(self.N), self.wires)
        
        @qml.qnode(self.device, interface="jax")
        def circuit_drawer(self):
            self.encoder_circuit_fun(np.arange(self.n_params))
            
            return [qml.expval(qml.PauliZ(int(k))) for k in self.wires_trash]
            
        self.drawer = qml.draw(circuit_drawer)(self)
        
    def vqe_enc_circuit(self, vqe_p, qcnn_p):
        self.vqe.circuit(vqe_p)
        self.encoder_circuit_fun(qcnn_p)

    def psi_enc_circuit(self, psi, qcnn_p):
        qml.QubitStateVector(psi, wires=[int(k) for k in range(self.N)])
        self.encoder_circuit_fun(qcnn_p)
        
    def train(self, lr, n_epochs, train_index, circuit=False, plot=False, inject=False):
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
            print(self.drawer)

        if not inject:
            X_train = jnp.array(self.vqe_params0[train_index])

            @qml.qnode(self.device, interface="jax")
            def q_encoder_circuit(vqe_params, params):
                self.vqe_enc_circuit(vqe_params, params)
                
                # return <psi|H|psi>
                return [qml.expval(qml.PauliZ(int(k))) for k in self.wires_trash]
        else:
            psi = []
            for h in self.vqe.Hs.mat_Hs:
                # Compute eigenvalues and eigenvectors
                eigval, eigvec = jnp.linalg.eigh(h)
                # Get the eigenstate to the lowest eigenvalue
                gstate = eigvec[:,jnp.argmin(eigval)]

                psi.append(gstate)
            psi = jnp.array(psi)
            self.psi = psi
            
            X_train = jnp.array(psi[train_index]) 

            @qml.qnode(self.device, interface="jax")
            def q_encoder_circuit(psi, params):
                self.psi_enc_circuit(psi, params)
                
                # return <psi|H|psi>
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
        get_compression = jax.jit(
            lambda p: jnp.sum(v_q_encoder_circuit(p, X_train), axis=1)
            / len(self.wires_trash)
        )
        
        def update(params, opt_state):
            grads = jd_compress(params)
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
                loss.append(j_compress(params))
                progress.set_description("Cost: {0}".format(loss[-1]))
            progress.update(1)

        self.params = params
        self.train_index = train_index

        if plot:
            plt.title("Loss of the encoder")
            plt.plot(np.arange(len(loss)) * 100, loss)

    def show_compression_isingchain(self, train_index=False):
        '''
        Shows result of compression of the Anomaly Detector
        '''
        train_index = self.train_index

        X_train = jnp.array(self.vqe_states[train_index])
        test_index = np.setdiff1d(np.arange(len(self.vqe_states)), train_index)
        X_test = jnp.array(self.vqe_states[test_index])

        @qml.qnode(self.device, interface="jax")
        def encoder_circuit_ic(vqe_params, params):
            self.circuit(vqe_params, params)

            # return <psi|H|psi>
            return [qml.expval(qml.PauliZ(int(k))) for k in self.wires_trash]

        v_encoder_circuit = jax.vmap(lambda x: encoder_circuit_ic(x, self.params))

        exps_train = (1 - np.sum(v_encoder_circuit(X_train), axis=1) / 4) / 2
        exps_test = (1 - np.sum(v_encoder_circuit(X_test), axis=1) / 4) / 2

        plt.figure(figsize=(10, 3))
        plt.scatter(train_index, exps_train)
        plt.scatter(
            np.setdiff1d(np.arange(len(X_train) + len(X_test)), train_index),
            exps_test,
            label="Test",
        )
        plt.axvline(x=len(self.vqe_states) // 2, color="red", linestyle="--")
        plt.legend()
        plt.grid(True)

    def save(filename):
        """
        Saves Encoder parameters to file

        Parameters
        ----------
        filename : str
            File where to save the parameters
        """

        things_to_save = [self.params, self.circuit_fun]

        with open(filename, "wb") as f:
            pickle.dump(things_to_save, f)


def load(filename_vqe, filename_enc):
    """
    Load Encoder from VQE file and Encoder file
    
    Parameters
    ----------
    filename_vqe : str
        Name of the file from where to load the VQE class
    filename_enc : str
        Name of the file from where to load the main parameters of the encoder class
    """
    loaded_vqe = vqe.load(filename_vqe)

    with open(filename_qcnn, "rb") as f:
        params, enc_circuit_fun = pickle.load(f)

    loaded_enc = encoder(vqe, enc_circuit_fun)
    loaded_enc.params = params

    return loaded_enc

def enc_classification_ANNNI(vqeclass, lr, epochs):
    """
    Train 3 encoder on the corners: 
    > K = 0, L = 2 (Paramagnetic)
    > K = 0, L = 0 (Ferromagnetic)
    > K = 1, L = 0 (Antiphase)
    The other states will be classified taking the lowest error in each encoder
    
    Parameters
    ----------
    vqeclass : class
        VQE class
    lr : float
        Learning rate for each training
    epochs : int
        Number of epochs for each training
    inject : bool
        If true, uses the real groundstates as inputs
        
    Returns
    -------
    np.ndarray
        Array of labels
    """
    # indexes of the 3 corner points
    side = int(np.sqrt(vqeclass.n_states))
    phase1 = 0
    phase2 = side - 1
    phase3 = int(vqeclass.n_states - side)
    
    encclass  = encoder(vqeclass, encoder_circuit)
    
    X = jnp.array(encclass.vqe_params0)

    @qml.qnode(encclass.device, interface="jax")
    def encoder_circuit_class(vqe_params, params):
        encclass.vqe_enc_circuit(vqe_params, params)

        return [qml.expval(qml.PauliZ(int(k))) for k in encclass.wires_trash]
    
    encoding_scores = []
    
    for phase in [phase1,phase2,phase3]:
        encclass  = encoder(vqeclass, encoder_circuit)
        encclass.train(lr, epochs, np.array([phase]), circuit = False, plot = False, inject = inject)
        v_encoder_circuit = jax.vmap(lambda x: encoder_circuit_class(x, encclass.params))
        exps = (1 - np.sum(v_encoder_circuit(X), axis=1) / 4) / 2
        exps = np.rot90( np.reshape(exps, (side,side)) )
        
        encoding_scores.append(exps)
        
    def getlines(func, xrange, side, color, res = 100):
        xs = np.linspace(xrange[0], xrange[1], res)
        ys = func(xs)
        plt.plot(side*xs -.5, side - ys*side/2 -.5, color = color, alpha=.8)

    def B2SA(x):
        return 1.05 * np.sqrt((x-.5)*(x-.1))

    def ferropara(x):
        return 1 - 2*x
        
    getlines(B2SA, [.5,1], side, 'white', res = 100)
    getlines(ferropara, [0,.5], side, 'white', res = 100)

    phases = mpl.colors.ListedColormap(["navy", "crimson", "limegreen", "limegreen"])
    norm = mpl.colors.BoundaryNorm(np.arange(0,4), phases.N) 
    plt.imshow(np.argmin(np.array(encoding_scores), axis = (0)), cmap = phases, norm = norm)

    plt.ylabel(r'$h$', fontsize=24)
    plt.xlabel(r'$\kappa$', fontsize=24)

    plt.xticks(ticks=np.linspace(0,side-1,5).astype(int), labels= [np.round(k*1/4,2) for k in range(0,5)], fontsize=18)
    plt.yticks(ticks=np.linspace(0,side-1,5).astype(int), labels= [np.round(k*2/4,2) for k in range(4,-1,-1)], fontsize=18)
        
    return np.argmin(np.array(encoding_scores), axis = (0))
