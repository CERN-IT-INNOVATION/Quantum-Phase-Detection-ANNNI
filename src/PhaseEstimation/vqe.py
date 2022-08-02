""" This module implements the base function to implement a VQE for a Ising Chain. """
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.example_libraries import optimizers

import copy
from tqdm.auto import tqdm
import joblib, pickle  # Writing and loading

import warnings

warnings.filterwarnings(
    "ignore",
    message="For Hamiltonians, the eigenvalues will be computed numerically. This may be computationally intensive for a large number of wires.Consider using a sparse representation of the Hamiltonian with qml.SparseHamiltonian.",
)

import sys, os
sys.path.insert(0, '../../')
import PhaseEstimation.circuits as circuits
import PhaseEstimation.losses as losses
import PhaseEstimation.general as qmlgen

##############

def circuit_ising(N, params):
    """
    Full VQE circuit

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
    for _ in range(6):
        index = circuits.circuit_ID9(active_wires, params, index)
        qml.Barrier()
        
    index = circuits.wall_gate(active_wires, qml.RX, params, index)
    
    return index

class vqe:
    def __init__(self, Hs, circuit):
        """
        Class for the VQE algorithm

        Parameters
        ----------
        Hs : class
            Custom Hamiltonian class
        circuit : function
            Function of the VQE circuit
        """
        self.N = Hs.N
        self.Hs = Hs
        self.n_states = Hs.n_states
        self.circuit = lambda p: circuit(self.N, p)
        self.n_params = self.circuit([0] * 10000)
        self.vqe_params0 = jnp.array( np.random.uniform(-np.pi, np.pi, size=(self.n_states,self.n_params)) ) 
        self.device = qml.device("default.qubit.jax", wires=self.N, shots=None)
        self.states_dist = []
        self.circuit_fun = circuit
        
        @qml.qnode(self.device, interface="jax")
        def vqe_state(self):
            self.circuit(np.arange(self.n_params))

            return qml.state()
        
        self.drawer = qml.draw(vqe_state)(self)
        
        ### STATES FUNCTIONS ###
        # Quantum Circuit to output states
        @qml.qnode(self.device, interface="jax")
        def q_vqe_state(vqe_params):
            self.circuit(vqe_params)

            return qml.state()

        self.v_q_vqe_state = jax.vmap(lambda v: q_vqe_state(v), in_axes=(0))  # vmap of the state circuit
        self.jv_q_vqe_state = jax.jit(self.v_q_vqe_state)                          # jitted vmap of the state circuit
        self.j_q_vqe_state = jax.jit(lambda p: q_vqe_state(p))                # jitted state circuit

        # For updating progress bar on fidelity between true states and vqe states
        self.jv_fidelties = jax.jit(lambda true, pars: losses.vqe_fidelities(true, pars, q_vqe_state) )
        
        def compute_true_state(H):
            eigval, eigvec = jnp.linalg.eigh(H)   # Get eigenvalues/-vectors of H
            gstate = eigvec[:,jnp.argmin(eigval)] # Pick the eigenvector with the lowest eigenvalue
            return gstate

        self.jv_compute_true_state = jax.jit(jax.vmap(compute_true_state))

        ### ENERGY FUNCTIONS ###
        # computes <psi|H|psi>
        def compute_vqe_E(state, Hmat):
            return jnp.real(jnp.conj(state) @ Hmat @ state)

        self.j_compute_vqe_E = jax.jit(compute_vqe_E)
        self.v_compute_vqe_E = jax.vmap(compute_vqe_E, in_axes=(0, 0))
        self.jv_compute_vqe_E = jax.jit(self.v_compute_vqe_E)
        
        def loss(params, Hs):
            pred_states = self.v_q_vqe_state(params)
            vqe_e = self.v_compute_vqe_E(pred_states, Hs)
            
            return jnp.mean(jnp.real(vqe_e))

        # Grad function, used in updating the parameters
        self.jd_loss = jax.jit(jax.grad(loss))
        
    def update(self, params, Hs_batch, opt_state, opt_update, get_params):
        grads = self.jd_loss(params, Hs_batch)
        opt_state = opt_update(0, grads, opt_state)

        return get_params(opt_state), opt_state 
        
    def train(self, lr, n_epochs, circuit=False, recycle = False, epochs_batch_size = 500, batch_size = 100):
        """
        Training function for the VQE.

        Parameters
        ----------
        lr : float
            Learning rate to be multiplied in the circuit-gradient output
        n_epochs : int
            Total number of epochs for each learning
        reg : float
            Regularizer of the training function. It ensures the subsequent states are not
            too much different
        circuit : bool
            if True -> Prints the circuit
        plots : bool
            if True -> Display plots
        recycle : bool
            if True -> Each state (depending on the intensity of the magnetic field) is computed independently and in parallel.
            if False -> Each state is trained after the previous one, in which the initial parameters are the final learnt parameters of the previous state
        """
        # IDEA: Making the Jax version of a VQE eigensolver is a bit less intuitive that in the QCNN learning function,
        #       since here we have l_steps different circuits since each output of the circuit is <psi|H|psi> where H,
        #       changes for each datapoint.
        #       Here the output of each circuit is |psi>, while the Hs is moved to the loss function
        #       <PSI|Hs|PSI> is computed through two jax.einsum

        if circuit:
            # Display the circuit
            print("+--- CIRCUIT ---+")
            print(self.drawer)
        
        # Initialize vqe arrays 
        params = copy.copy(self.vqe_params0)    # Gate rotations of the circuit
        MSE    = []                            # Losses
        vqe_E  = np.array([0.0]*self.n_states) # Array of energies (VQE)
        true_E = np.array([0.0]*self.n_states) # Array of energies (true)
        
        if recycle == True and batch_size == 1:
            previous_params = jnp.array( np.random.uniform(-np.pi, np.pi, size=(1,self.n_params)) ) 

        # Prepare array of indices for batch-training
        batches = []
        k = 0
        while k < self.n_states:
            batches.append(self.Hs.recycle_rule[k:k+batch_size])
            k = k + batch_size
        n_batches = len(batches)

        progress = tqdm(range(n_batches), position=0, leave=True)
        for batch_number, batch in enumerate(batches):
            params_batch = params[batch] # Array of rotations for the VQE gate rotations
                                         # of the indices in batches
            Hs_batch     = []            # Array of Hamiltonians (in matricial form) to 
                                         # compute the energies to minimize
            psi_batch    = []            # Array of true GS to compute fidelity
            MSE_batch    = []
            
            if recycle == True and batch_size == 1:
                params_batch = previous_params

            # Prepare training of the batch computing Hs and psis
            for idx in batch:
                H, en, psi  = qmlgen.get_H_eigval_eigvec(self.Hs.qml_Hs[idx], 0)
                true_E[idx] = en 
                Hs_batch.append(H)
                psi_batch.append(psi)
            psi_batch = jnp.array(psi_batch)
            Hs_batch = jnp.array(Hs_batch)

            # Defining an optimizer in Jax
            opt_init, opt_update, get_params = optimizers.adam(lr)
            opt_state = opt_init(params_batch)

            epochs = n_epochs
            if batch_number == 0 and recycle == True and batch_size == 1:
                epochs = 10*n_epochs
            for it in range(epochs):
                params_batch, opt_state = self.update(params_batch, Hs_batch, opt_state, opt_update, get_params)

                # skip when it == 0
                if (it + 1) % epochs_batch_size == 0:
                    pred_states = self.jv_q_vqe_state(params_batch)
                    MSE_batch.append(
                        jnp.mean(jnp.square(self.jv_compute_vqe_E(pred_states, Hs_batch) - true_E[batch]))
                    )

                    # Update progress bar
                    progress.set_description("Batch# {0}/{1} | IDX(Batch) {2}/{3} | Cost(Batch): {4:.5f} | Mean F.(Batch): {5:.5f}".format(
                                             batch_number, n_batches, it, n_epochs,
                                             MSE_batch[-1], self.jv_fidelties(psi_batch, jnp.array(params_batch)) ) )
                    
            # Save rotations and vqe energies found
            params[batch] = params_batch
            vqe_E[batch] = self.jv_compute_vqe_E(self.jv_q_vqe_state(params_batch), Hs_batch)
            MSE.append(MSE_batch)
            progress.update(1)
            
            if recycle == True and batch_size == 1:
                previous_params = params_batch
            
        MSE = np.mean(MSE, axis = 0)

        self.MSE0 = MSE
        self.vqe_params0 = params
        self.vqe_e0  = np.array(vqe_E)
        self.true_e0 = np.array(true_E)
        
    def train_site(self, lr, n_epochs, site):
        index = [site]
        H     = [ np.real(qml.matrix(self.Hs.qml_Hs[site])).astype(np.single) ]
        H     = jnp.array(H)
        param = copy.copy(self.vqe_params0[index])
        
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(param)
        
        for it in range(n_epochs):
            param, opt_state = self.update(param, H, opt_state, opt_update, get_params)
        
        self.vqe_e0[site]      = self.jv_compute_vqe_E(self.jv_q_vqe_state(param), H)
        self.vqe_params0[site] = param
        
        return jnp.square(self.jv_compute_vqe_E(self.jv_q_vqe_state(param), H) - self.true_e0[site])
    
    def train_refine(self, lr, n_epochs, acc_thr, assist = False):
        progress = tqdm(range(self.n_states), position=0, leave=True)
        for site in self.Hs.recycle_rule:
            accuracy = np.abs((self.vqe_e0[site] - self.true_e0[site])/self.true_e0[site])

            if accuracy > acc_thr:
                if assist:
                    neighbours = np.array(qmlgen.get_neighbours(self, site))
                    neighbours_accuracies = np.abs((self.vqe_e0[neighbours] - self.true_e0[neighbours])/self.true_e0[neighbours])
                    best_neighbour = neighbours[np.argmin(neighbours_accuracies)]
                    self.vqe_params0[site] = copy.copy(self.vqe_params0[best_neighbour])
                self.train_site(lr, n_epochs, int(site) )

            progress.update(1)
          
    def train_site_excited(self, lr, n_epochs, site, beta):
        # Get all the necessary training parameters for the VQD algorithm
        # > H: Hamiltonian of the model
        # > H_eff: Effective Hamiltonian for the model (H +|psi><psi|)
        # > site: index for (L,K) combination
        H, H_eff, self.true_e1[site] = qmlgen.get_VQD_params(self.Hs.qml_Hs[site], beta)
        
        param = copy.copy(self.vqe_params1[[site]])
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(param)
        for it in range(n_epochs):
            param, opt_state = self.update(param, H_eff, opt_state, opt_update, get_params)
            
        self.vqe_e1[site]      = self.jv_compute_vqe_E(self.jv_q_vqe_state(param), H)
        self.vqe_params1[site] = param
        
    def train_excited(self, lr, n_epochs, beta):
        self.vqe_e1, self.vqe_params1, self.true_e1 = np.zeros((self.n_states,)), np.zeros((self.n_states,self.n_params)), np.zeros((self.n_states,))
        
        progress = tqdm(self.Hs.recycle_rule, position=0, leave=True)
        for site in progress:
            epochs = 10*n_epochs if site == 0 else n_epochs
            if site == 0:
                epochs = 10*n_epochs
                self.vqe_params1[site] = jnp.array( np.random.uniform(-np.pi, np.pi, size=(self.n_params)) ) 
            else:
                epochs = n_epochs
                self.vqe_params1[site] = copy.copy(self.vqe_params1[pred_site])
                
            self.train_site_excited(lr, epochs, int(site), beta)
            pred_site = site
            
    def train_refine_excited(self, lr, n_epochs, acc_thr, beta, assist = False):
        progress = tqdm(range(self.n_states), position=0, leave=True)
        for site in self.Hs.recycle_rule:
            accuracy = np.abs((self.vqe_e1[site] - self.true_e1[site])/self.true_e1[site])

            if accuracy > acc_thr:
                if assist:
                    neighbours = np.array(qmlgen.get_neighbours(self, site))
                    neighbours_accuracies = np.abs((self.vqe_e1[neighbours] - self.true_e1[neighbours])/self.true_e1[neighbours])
                    best_neighbour = neighbours[np.argmin(neighbours_accuracies)]
                    self.vqe_params1[site] = copy.copy(self.vqe_params1[best_neighbour])
                self.train_site_excited(lr, n_epochs, int(site), beta)

            progress.update(1)

    def save(self, filename):
        """
        Save main parameters of the VQE class to a local file.
        Parameters saved:
        > Hs class, vqe parameters, circuit function

        Parameters
        ----------
        filename : str
            Local file to save the parameters
        """
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


def load_vqe(filename):
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
