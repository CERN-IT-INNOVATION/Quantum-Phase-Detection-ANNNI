""" This module implements the base function to implement a VQE for a Ising Chain. """
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.example_libraries import optimizers

import copy
import tqdm  # Pretty progress bars
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
        self.vqe_params = jnp.array( np.random.uniform(-np.pi, np.pi, size=(self.n_states,self.n_params)) ) 
        self.device = qml.device("default.qubit.jax", wires=self.N, shots=None)
        self.MSE = []
        self.vqe_e = []
        self.recycle = False
        self.states_dist = []

        self.circuit_fun = circuit
        
        @qml.qnode(self.device, interface="jax")
        def vqe_state(self):
            self.circuit(np.arange(self.n_params))

            return qml.state()

        self.drawer = qml.draw(vqe_state)(self)

    def train(self, lr, n_epochs, reg=0, circuit=False, recycle=True, epochs_batch_size = 500, excited = False, beta = 0):
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
            if False -> It does not display the circuit
        plots : bool
            if True -> Display plots
            if False -> It does not display plots
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
        
        # Quantum Circuit to output states
        @qml.qnode(self.device, interface="jax")
        def vqe_state(vqe_params):
            self.circuit(vqe_params)

            return qml.state()
        
        # vmap of the circuit
        v_vqe_state = jax.vmap(lambda v: vqe_state(v), in_axes=(0))

        # jitted vmap of the circuit
        jv_vqe_state = jax.jit(v_vqe_state)

        # jitted circuit
        j_vqe_state = jax.jit(lambda p: vqe_state(p))
        
        def compute_true_state(H):
            # Compute eigenvalues and eigenvectors
            eigval, eigvec = jnp.linalg.eigh(H)
            # Get the eigenstate to the lowest eigenvalue
            gstate = eigvec[:,jnp.argmin(eigval)]

            return gstate
            
        jv_compute_true_state = jax.jit(jax.vmap(compute_true_state))
        
        # Same function as above but returns the energies not MSE
        def compute_vqe_E(state, Hmat):
            vqe_e = jnp.conj(state) @ Hmat @ state

            return jnp.real(vqe_e)

        j_compute_vqe_E = jax.jit(compute_vqe_E)
        v_compute_vqe_E = jax.vmap(compute_vqe_E, in_axes=(0, 0))
        jv_compute_vqe_E = jax.jit(v_compute_vqe_E)
        
        if excited:
            # Compute outer product between a state and itself (VQD)
            def psi_outer(psi):
                return jnp.outer(jnp.conj(psi), psi)
            # Compute outer product of all ground states
            outers = jax.vmap(psi_outer)(self.states)
            
            # Try to load already trained parameters for
            # the excited states, if not, generate random ones
            try:
                params = copy.copy(self.vqe_params1)
            except:
                params = jnp.array( np.random.uniform(-np.pi, np.pi, size=(self.n_states,self.n_params)) )
                
            operators = self.Hs.mat_Hs + beta * outers
                
        else:
            self.true_states = jv_compute_true_state(self.Hs.mat_Hs)
            params = copy.copy(self.vqe_params)
            operators = self.Hs.mat_Hs
            true_e = self.Hs.true_e
        
        # Check if the VQE class already has computed 
        # the true ground states
        try:
            self.gs_states
        except:
            self.gs_states = jv_compute_true_state(self.Hs.mat_Hs)

        if not recycle:
            self.recycle = False
            jv_fidelties = jax.jit(lambda true, pars: losses.vqe_fidelities(true, pars, vqe_state) )
            
            def compute_diff_states(states):
                return jnp.mean(jnp.square(jnp.diff(jnp.real(states), axis=1)))
            
            def loss(params):
                pred_states = v_vqe_state(params)
                vqe_e = v_compute_vqe_E(pred_states, operators)

                if reg != 0:
                    return jnp.mean(jnp.real(vqe_e)) + reg * compute_diff_states(pred_states)
                else:
                    return jnp.mean(jnp.real(vqe_e))

            # Grad function, used in updating the parameters
            jd_loss = jax.jit(jax.grad(loss))
            
            def update(params, opt_state):
                grads = jd_loss(params)
                opt_state = opt_update(0, grads, opt_state)
                
                return get_params(opt_state), opt_state 

            # Defining an optimizer in Jax
            opt_init, opt_update, get_params = optimizers.adam(lr)
            opt_state = opt_init(params)
            
            MSE = []
            
            progress = tqdm.tqdm(range(n_epochs), position=0, leave=True)
            for it in progress:
                params, opt_state = update(params, opt_state)
                
                # I want to skip when it == 0
                if (it + 1) % epochs_batch_size == 0:
                    pred_states = jv_vqe_state(params)
                    MSE.append(
                        jnp.mean(jnp.square(jv_compute_vqe_E(pred_states, operators) - true_e))
                    )

                    # Update progress bar
                    progress.set_description("Cost: {0:.4f} | Mean F.: {1:.4f}".format(MSE[-1], jv_fidelties(self.true_states, jnp.array(params)) ) )
                    
        else:
            self.recycle = True
            
            def loss_reg(param, Hmat, reg, previous_state):
                pred_state = j_vqe_state(param)
                vqe_e = j_compute_vqe_E(pred_state, Hmat)
                
                return jnp.real(vqe_e) + reg * jnp.square(jnp.abs(jnp.conj(pred_state) @  previous_state))
            
            def loss(param, Hmat):
                pred_state = j_vqe_state(param)
                vqe_e = j_compute_vqe_E(pred_state, Hmat)
                
                return jnp.real(vqe_e)
    
            # Grad function of the MSE, used in updating the parameters
            jd_loss_reg = jax.jit(jax.grad(loss_reg))
            jd_loss = jax.jit(jax.grad(loss))
            
            def update(param, opt_state, Hmat):
                grads = jd_loss(param, Hmat)
                opt_state = opt_update(0, grads, opt_state)
                
                return get_params(opt_state), opt_state 
            
            def update_reg(param, opt_state, Hmat, reg, previous_state):
                grads = jd_loss_reg(param, Hmat, reg, previous_state)
                opt_state = opt_update(0, grads, opt_state)
                
                return get_params(opt_state), opt_state 

            progress = tqdm.tqdm(self.Hs.recycle_rule[1:], position=0, leave=True)
            params = [[0]*self.n_params]*self.n_states
            param = jnp.array(np.random.rand(self.n_params))
            
            MSE = []
            previous_pred_states = []
            
            idx = 0
            
            # Defining an optimizer in Jax
            opt_init, opt_update, get_params = optimizers.adam(lr)
            opt_state = opt_init(param)
            
            for it in range(10*n_epochs):
                param, opt_state = update(param, opt_state, operators[idx])
                
            previous_state = j_vqe_state(param)
            params[idx] = copy.copy(param)
            progress.set_description("{0}/{1}".format(idx + 1, self.n_states))
            
            for idx in (progress):
                opt_init, opt_update, get_params = optimizers.adam(lr)
                opt_state = opt_init(param)
                
                for it in range(n_epochs):
                    param, opt_state = update_reg(param, opt_state, operators[idx], reg, previous_state)
                        
                params[idx] = copy.copy(param)
                progress.set_description("{0}/{1}".format(idx + 1, self.n_states))
                previous_state = j_vqe_state(param)
            params = jnp.array(params)
        
        if not excited: 
            self.MSE = MSE
            self.vqe_params = params
            self.states = jv_vqe_state(params)
            self.vqe_e = jv_compute_vqe_E(self.states, self.Hs.mat_Hs)
            
        else:
            self.MSE1 = MSE
            self.vqe_params1 = params
            self.states1 = jv_vqe_state(params)   
            self.vqe_e1 = jv_compute_vqe_E(self.states1, self.Hs.mat_Hs) 

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
        things_to_save = [
            self.Hs,
            self.vqe_params,
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

    Hs, vqe_params, circuit_fun = things_to_load

    loaded_vqe = vqe(Hs, circuit_fun)
    loaded_vqe.vqe_params = vqe_params
    
    return loaded_vqe
