""" This module implements the base function to implement a VQE for a Ising Chain. """
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

import copy
from tqdm.auto import tqdm
import pickle  # Writing and loading

import warnings
warnings.filterwarnings(
    "ignore",
    message="For Hamiltonians, the eigenvalues will be computed numerically. This may be computationally intensive for a large number of wires.Consider using a sparse representation of the Hamiltonian with qml.SparseHamiltonian.",
)

from PhaseEstimation import circuits, losses, hamiltonians
from PhaseEstimation import general as qmlgen

from typing import List, Callable
from numbers import Number

##############

def circuit_ising(N : int, params : List[Number]) -> int:
    """
    Full VQE circuit
    Number of parameters (gates): 7*N
    Parameters
    ----------
    N : int
        Number of qubits
    params: np.ndarray
        Array of parameters/rotation for the circuit

    Returns
    -------
    int
        Total number of parameters needed to build this circuit
    """
    # No wire will be measured until the end, the array of active
    # wire will correspond to np.arange(N) throughout the whole circuit
    active_wires = np.arange(N)
    index = 0
    qml.Barrier()
    for _ in range(6): # Depth of the circuit
        # Iterate circuit_ID9, the deeper the merrier
        index = circuits.circuit_ID9(active_wires, params, index)
        qml.Barrier()
    
    # Final independent rotations RX for each wire
    index = circuits.wall_gate(active_wires, qml.RX, params, index)
    
    return index

def circuit_ising2(N : int, params : List[Number]) -> int:
    """
    Full VQE circuit, enhanced version of circuit_ising, higher number of parameters
    Number of parameters (gates): 11*N

    Parameters
    ----------
    N : int
        Number of qubits
    params: np.ndarray
        Array of parameters/rotation for the circuit

    Returns
    -------
    int
        Total number of parameters needed to build this circuit
    """
    # No wire will be measured until the end, the array of active
    # wire will correspont to np.arange(N) throughout the whole circuit
    active_wires = np.arange(N)
    index = 0
    qml.Barrier()
    for _ in range(9):
        index = circuits.circuit_ID9(active_wires, params, index)
        qml.Barrier()
        
    index = circuits.wall_gate(active_wires, qml.RX, params, index)
    index = circuits.wall_gate(active_wires, qml.RY, params, index)
    
    return index

class vqe:
    def __init__(self, Hs : hamiltonians.hamiltonian, circuit : Callable):
        """
        Class for the VQE algorithm

        Parameters
        ----------
        Hs : hamiltonians.hamiltonian
            Custom Hamiltonian class
        circuit : function
            Function of the VQE circuit
        """
        self.Hs = Hs
        self.circuit = lambda p: circuit(self.Hs.N, p)
        self.circuit_fun = circuit
        # Pass the parameter array [0]*10000 (intentionally large) to the circuit
        # which it will output `index`, namely the number of parameters
        self.n_params = self.circuit([0] * 10000)
        # Initialize randomly all the parameter-arrays for each state
        self.vqe_params0 = jnp.array( np.random.uniform(-np.pi, np.pi, size=(self.Hs.n_states,self.n_params)) )
        self.device = qml.device("default.qubit.jax", wires=self.Hs.N, shots=None)

        ### STATES FUNCTIONS ###
        # QCircuit: CIRCUIT(params) -> PSI
        @qml.qnode(self.device, interface="jax")
        def q_vqe_state(vqe_params):
            self.circuit(vqe_params)

            return qml.state()

        self.v_q_vqe_state = jax.vmap(lambda v: q_vqe_state(v), in_axes=(0))  # vmap of the state circuit
        self.jv_q_vqe_state = jax.jit(self.v_q_vqe_state)                     # jitted vmap of the state circuit
        self.j_q_vqe_state = jax.jit(lambda p: q_vqe_state(p))                # jitted state circuit

        # For updating progress bar on fidelity between true states and vqe states
        self.jv_fidelties = jax.jit(lambda true, pars: losses.vqe_fidelities(true, pars, q_vqe_state) )

        ### ENERGY FUNCTIONS ###
        # Computes <psi|H|psi>
        def compute_vqe_E(state, Hmat):
            return jnp.real(jnp.conj(state) @ Hmat @ state)

        self.j_compute_vqe_E = jax.jit(compute_vqe_E)
        self.v_compute_vqe_E = jax.vmap(compute_vqe_E, in_axes=(0, 0))
        self.jv_compute_vqe_E = jax.jit(self.v_compute_vqe_E)
        
        # Loss function: LOSS = 1/n_states SUM_i ( ENERGY(psi_i) )
        def loss(params, Hs):
            pred_states = self.v_q_vqe_state(params)
            vqe_e = self.v_compute_vqe_E(pred_states, Hs)
            
            # Cast as real because energies are supposed to be it
            return jnp.mean(jnp.real(vqe_e))

        # Grad function, used in updating the parameters
        self.jd_loss = jax.jit(jax.grad(loss))

    def __repr__(self):
        # QCircuit just for printing it
        @qml.qnode(self.device, interface="jax")
        def vqe_state(self):
            # Passing np.arange array for enumerating the parameters
            self.circuit(np.arange(self.n_params))

            return qml.state()

        return qml.draw(vqe_state)(self)
        
    def _update(self, params, Hs_batch, opt_state, opt_update, get_params):
        grads = self.jd_loss(params, Hs_batch)
        opt_state = opt_update(0, grads, opt_state)

        return get_params(opt_state), opt_state 
        
    def train_site(self, lr : Number, n_epochs : int, site : int):
        """
        Minimize <psi|H|psi> for a single site
        
        """
        # Get all the necessary training parameters for the VQD algorithm
        # > H: Hamiltonian of the model
        # > H_eff: Effective Hamiltonian for the model (H +|psi><psi|)
        # > site: index for (L,K) combination
        H, self.true_e0[site] = qmlgen.get_VQE_params(self.Hs.qml_Hs[site])

        index = [site]
        param = copy.copy(self.vqe_params0[index])
        
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(param)
        
        for _ in range(n_epochs):
            param, opt_state = self._update(param, H, opt_state, opt_update, get_params)
        
        self.vqe_e0[site]      = self.jv_compute_vqe_E(self.jv_q_vqe_state(param), H)
        self.vqe_params0[site] = param
          
    def train_site_excited(self, lr : Number, n_epochs : int, site : int, beta : Number):
        """
        Minimize <psi|H|psi> + beta|<psi|psi_0>|^2 for a single site
        """
        # Get all the necessary training parameters for the VQD algorithm
        # > H: Hamiltonian of the model
        # > H_eff: Effective Hamiltonian for the model (H +|psi><psi|)
        # > site: index for (L,K) combination
        H, H_eff, self.true_e1[site] = qmlgen.get_VQD_params(self.Hs.qml_Hs[site], beta)
        
        param = copy.copy(self.vqe_params1[[site]])
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(param)
        for _ in range(n_epochs):
            param, opt_state = self._update(param, H_eff, opt_state, opt_update, get_params)
            
        self.vqe_e1[site]      = self.jv_compute_vqe_E(self.jv_q_vqe_state(param), H)
        self.vqe_params1[site] = param

    def train(self, lr : Number, n_epochs : int, circuit : bool = False):
        """
        Training function for the VQE.

        Parameters
        ----------
        lr : float
            Learning rate to be multiplied in the circuit-gradient output
        n_epochs : int
            Total number of epochs for each learning
        circuit : bool
            if True -> Prints the circuit
        """
        pred_site : int

        if circuit:
            # Display the circuit
            print("+--- CIRCUIT ---+")
            print(self)
        
        # The true GS energies will be computed during training
        # and not during initialization of the VQE since it 
        # requires the diagonalization of many large matrices
        self.vqe_e0, self.vqe_params0, self.true_e0 = np.zeros((self.Hs.n_states,)), np.zeros((self.Hs.n_states,self.n_params)), np.zeros((self.Hs.n_states,))
        
        progress = tqdm(self.Hs.recycle_rule, position=0, leave=True)
        # Site will follow the order of Hs.recycle rule:
        # For ANNI Model:
        #     INDICES              RECYCLE RULE
        # +--------------+       +--------------+
        # | 4  9  14  19 |       | 4  5  14  15 |
        # | 3  8  13  18 |       | 3  6  13  16 |
        # | 2  7  12  17 |  ==>  | 2  7  12  17 |
        # | 1  6  11  16 |       | 1  8  11  18 |
        # | 0  5  10  15 |       | 0  9  10  19 |
        # +--------------+       +--------------+
        #
        # For Ising Chain:
        #    INDICES               RECYCLE RULE
        # +------------ -        +------------ -
        # | 0 | 1 | 2 |     ==>  | 0 | 1 | 2 | 
        # +------------ -        +------------ -
        for site in progress:
            # First site will be trained more since it starts from a 
            # random configuration of parameters
            if site == 0:
                epochs = 10*n_epochs
                # Random initial state
                self.vqe_params0[site] = jnp.array( np.random.uniform(-np.pi, np.pi, size=(self.n_params)) ) 
            else:
                epochs = n_epochs
                # Initial state is the final state of last site trained
                self.vqe_params0[site] = copy.copy(self.vqe_params0[pred_site])
                
            self.train_site(lr, epochs, int(site))
            self.true_e0[site] 
            pred_site = site  # Previous site for next training

    def train_excited(self, lr : Number, n_epochs : int, beta : Number, circuit : bool = False):
        """
        Training function through VQD.
        VQD is trained finding psi such that:
        <psi|H|psi> + beta*|<psi_0|psi>|**2
        is minimized.

        Parameters
        ----------
        lr : float
            Learning rate to be multiplied in the circuit-gradient output
        n_epochs : int
            Total number of epochs for each learning
        beta : float
            Importance (regularization strenght) given to the orthogonalization with psi_0
        circuit : bool
            if True -> Prints the circuit
        """
        pred_site : int
        
        if circuit:
            # Display the circuit
            print("+--- CIRCUIT ---+")
            print(self)

        # Initialize parameters
        self.vqe_e1, self.vqe_params1, self.true_e1 = np.zeros((self.Hs.n_states,)), np.zeros((self.Hs.n_states,self.n_params)), np.zeros((self.Hs.n_states,))
        
        progress = tqdm(self.Hs.recycle_rule, position=0, leave=True)
        for site in progress:
            # First site will be trained more since it starts from a 
            # random configuration of parameters
            if site == 0:
                epochs = 10*n_epochs
                # Random initial state
                self.vqe_params1[site] = jnp.array( np.random.uniform(-np.pi, np.pi, size=(self.n_params)) ) 
            else:
                epochs = n_epochs
                # Initial state is the final state of last site trained
                self.vqe_params1[site] = copy.copy(self.vqe_params1[pred_site])
                
            self.train_site_excited(lr, epochs, int(site), beta)
            pred_site = site # Previous site for next training

    def train_refine(self, lr : Number, n_epochs : int, acc_thr : Number, assist : bool = False):
        """
        Training only the sites that have an accuracy score worse (higher) than acc_thr

        Parameters
        ----------
        lr : float
            Learning rate to be multiplied in the circuit-gradient output
        n_epochs : int
            Total number of epochs for each learning
        acc_thr : float
            Accuracy threshold for which selecting the sites to train
        assist : bool
            if True -> Each site that will be trained will start from the neighbouring site
            that has the better accuracy
        """
        progress = tqdm(range(self.Hs.n_states), position=0, leave=True)

        # Select the sites to train based on their accuracy score
        for site in self.Hs.recycle_rule:
            # Accuracy value of the given site
            accuracy = np.abs((self.vqe_e0[site] - self.true_e0[site])/self.true_e0[site])

            if accuracy > acc_thr: # If the accuracy is bad (higher than threshold)...
                # if assist we copy the state from the best neighbouring site and
                # starting training from there
                if assist:
                    # Array of indexes of neighbouring sites
                    neighbours = np.array(qmlgen.get_neighbours(self, site))
                    # Array of their respective accuracies
                    neighbours_accuracies = np.abs((self.vqe_e0[neighbours] - self.true_e0[neighbours])/self.true_e0[neighbours])
                    # Select the index of the neighbour with the best (lowest) accuracy score
                    best_neighbour = neighbours[np.argmin(neighbours_accuracies)]
                    self.vqe_params0[site] = copy.copy(self.vqe_params0[best_neighbour])
                # Start training the site
                self.train_site(lr, n_epochs, int(site) )

            progress.update(1)
            
    def train_refine_excited(self, lr : Number, n_epochs : int, acc_thr : Number, beta : Number, assist : bool = False):
        """
        Training only the sites that have an accuracy score worse (higher) than acc_thr
        for the excited states

        Parameters
        ----------
        lr : float
            Learning rate to be multiplied in the circuit-gradient output
        n_epochs : int
            Total number of epochs for each learning
        acc_thr : float
            Accuracy threshold for which selecting the sites to train
        assist : bool
            if True -> Each site that will be trained will start from the neighbouring site
            that has the better accuracy
        """
        progress = tqdm(range(self.Hs.n_states), position=0, leave=True)
        # Select the sites to train based on their accuracy score
        for site in self.Hs.recycle_rule:
            # Accuracy value of the given site
            accuracy = np.abs((self.vqe_e1[site] - self.true_e1[site])/self.true_e1[site])

            if accuracy > acc_thr: # If the accuracy is bad (higher than threshold)...
                # if assist we copy the state from the best neighbouring site and
                # starting training from there
                if assist:
                    # Array of indexes of neighbouring sites
                    neighbours = np.array(qmlgen.get_neighbours(self, site))
                    # Array of their respective accuracies
                    neighbours_accuracies = np.abs((self.vqe_e1[neighbours] - self.true_e1[neighbours])/self.true_e1[neighbours])
                    # Select the index of the neighbour with the best (lowest) accuracy score
                    best_neighbour = neighbours[np.argmin(neighbours_accuracies)]
                    self.vqe_params1[site] = copy.copy(self.vqe_params1[best_neighbour])
                # Start training the site
                self.train_site_excited(lr, n_epochs, int(site), beta)

            progress.update(1)
            
    def save(self, filename : str):
        """
        Save main parameters of the VQE class to a local file.
        Parameters saved:
        > Hs class, vqe parameters, circuit function

        Parameters
        ----------
        filename : str
            Local file to save the parameters
        """

        if not isinstance(filename, str):
            raise TypeError('Invalid name for file')

        # Check if the VQE was trained for excited states aswell:
        excited = True
        try:
            self.vqe_params1
        except:
            excited = False
            
        if not excited:
            things_to_save = [
                self.Hs,
                self.vqe_params0,
                self.vqe_e0,
                self.true_e0,
                self.circuit_fun
            ]
        else:
            things_to_save = [
                self.Hs,
                self.vqe_params0,
                self.vqe_e0,
                self.true_e0,
                self.vqe_params1,
                self.vqe_e1,
                self.true_e1,
                self.circuit_fun
            ]
            
        with open(filename, "wb") as f:
            pickle.dump(things_to_save, f)


def load_vqe(filename : str) -> vqe:
    """
    Load main parameters of a VQE class saved to a local file using vqe.save(filename)

    Parameters
    ----------
    filename : str
        Local file from where to load the parameters

    Returns
    -------
    class
        VQE class with main parameters
    """
    if not isinstance(filename, str):
            raise TypeError('Invalid name for file')

    with open(filename, "rb") as f:
        things_to_load = pickle.load(f)

    if len(things_to_load) == 5:
        Hs, vqe_params, vqe_e, true_e, circuit_fun = things_to_load
        loaded_vqe = vqe(Hs, circuit_fun)
    else:
        Hs, vqe_params, vqe_e, true_e, vqe_params1, vqe_e1, true_e1, circuit_fun = things_to_load
        loaded_vqe = vqe(Hs, circuit_fun)
        loaded_vqe.vqe_params1 = vqe_params1
        loaded_vqe.vqe_e1 = vqe_e1
        loaded_vqe.true_e1 = true_e1
    
    loaded_vqe.vqe_params0 = vqe_params
    loaded_vqe.vqe_e0 = vqe_e
    loaded_vqe.true_e0 = true_e
    
    return loaded_vqe
