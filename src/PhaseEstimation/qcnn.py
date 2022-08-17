""" This module implements the base functions to implement a VQE for a Ising Chain with Transverse Field. """
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

from matplotlib import pyplot as plt

import copy
import tqdm  # Pretty progress bars
import pickle  # Writing and loading

import warnings
warnings.filterwarnings(
    "ignore",
    message="For Hamiltonians, the eigenvalues will be computed numerically. This may be computationally intensive for a large number of wires.Consider using a sparse representation of the Hamiltonian with qml.SparseHamiltonian.",
)

from . import circuits as circuits
from . import general as qmlgen
from . import vqe as vqe
##############

def qcnn_circuit(params, N, n_outputs):
    """
    Building function for the QCNN circuit:
    
    Parameters
    ----------
    params : np.ndarray
        Array of QCNN parameters
    N : int
        Number of qubits
    n_outputs : int
        Output vector dimension

    Returns
    -------
    int
        Total number of parameters needed to build this circuit
    np.ndarray
        Array of indexes of not-measured wires (due to pooling)
    """

    # Wires that are not measured (through pooling)
    active_wires = np.arange(N)

    # Visual Separation VQE||QCNN
    qml.Barrier()
    qml.Barrier()

    # Index of the parameter vector
    index = 0
    
    # Iterate Convolution+Pooling until we only have a single wires
    index = circuits.wall_gate(active_wires, qml.RY, params, index)
    circuits.wall_cgate_serial(active_wires, qml.CNOT)
    while len(active_wires) > n_outputs:
        index = circuits.convolution(active_wires, params, index)
        index, active_wires = circuits.pooling(active_wires, qml.RX, params, index)
        
        qml.Barrier()
        
    circuits.wall_cgate_serial(active_wires, qml.CNOT)
    index = circuits.wall_gate(active_wires, qml.RY, params, index)

    # Return the number of parameters
    return index + 1, active_wires

class qcnn:
    def __init__(self, vqe, qcnn_circuit, n_outputs = 1):
        """
        Class for the QCNN algorithm

        Parameters
        ----------
        vqe : class
            VQE class
        qcnn_circuit :
            Function of the QCNN circuit
        n_outputs : int
            Output vector dimension
        """
        self.vqe = vqe
        self.N = vqe.N
        self.n_states = vqe.n_states
        self.n_outputs = n_outputs
        self.qcnn_circuit_fun = lambda p: qcnn_circuit(p, self.N, n_outputs)
        self.n_params, self.final_active_wires = self.qcnn_circuit_fun([0] * 10000)
        self.params = np.array(np.random.rand(self.n_params) )
        self.device = vqe.device

        self.vqe_params = np.array(vqe.vqe_params0)
        
        self.labels = np.array(vqe.Hs.labels)
        self.train_index = []
        self.loss_train = []
        self.loss_test = []

        @qml.qnode(self.device, interface="jax")
        def circuit_drawer(self):
            _, active_wires = self.qcnn_circuit_fun(np.arange(self.n_params))
            if n_outputs == 1:
                return qml.probs(wires=self.N - 1)
            else:
                return qml.probs([int(k) for k in self.final_active_wires])
            
        self.drawer = qml.draw(circuit_drawer)(self)
    
    def vqe_qcnn_circuit(self, vqe_p, qcnn_p):
        '''
        Circuit:
        VQE + QCNN
        '''
        self.vqe.circuit(vqe_p)
        self.qcnn_circuit_fun(qcnn_p)
    
    # Training function
    def train(self, lr, n_epochs, train_index, loss_fn, circuit=False, plot=False):
        """
        Training function for the QCNN.

        Parameters
        ----------
        lr : float
            Learning rate for the ADAM optimizer
        n_epochs : int
            Total number of epochs for each learning
        train_index : np.ndarray
            Index of training points
        loss_fn : function
            Loss function
        circuit : bool
            if True -> Prints the circuit
        plot : bool
            if True -> It displays loss curve
        """

        # None could be in the labels as [None, None] when training
        # ANNNI model which non-trivial cases have no solution
        if None not in self.labels:
            X_train, Y_train = jnp.array(self.vqe_params[train_index]), jnp.array(
                self.labels[train_index]
            )
            test_index = np.setdiff1d(np.arange(len(self.vqe_params)), train_index)
            X_test, Y_test = jnp.array(self.vqe_params[test_index]), jnp.array(
                self.labels[test_index]
            )
        else:
            # If we are traing an ANNNI model, we have to first restrict on the trivial cases:
            # L = 0, K = whatever
            # K = 0, L = whatever
            mask = jnp.array(jnp.logical_or(jnp.array(self.vqe.Hs.model_params)[:,1] == 0, jnp.array(self.vqe.Hs.model_params)[:,2] == 0))
            
            self.vqe_params = jnp.array(self.vqe_params)
            
            X, Y = self.vqe_params[mask], self.labels[mask,:].astype(int)
            # The labels stored in the Hamiltonian class are:
            #   > [1,1] for paramagnetic states
            #   > [0,1] for ferromagnetic states
            #   > [1,0] for antiphase states
            #   > [0,0] not used
            #   > [None,None] for states with no analytical solutions
            # qml.probs(wires = active_wires) will output the following probabilities:
            # (example for a two qbits output) 
            # p(00), p(01), p(10), p(11)
            # The labels need to be transformed accordingly
            #     [0,0] -> [1,0,0,0] trash case
            #     [0,1] -> [0,1,0,0] for paramagnetic
            #     [1,0] -> [0,0,1,0] for antiphase
            #     [1,1] -> [0,0,0,1] for ferromagnetic
            Ymix = []
            for label in Y:
                if (label == [0,0]).all():
                    Ymix.append([1,0,0,0])
                elif (label == [0,1]).all():
                    Ymix.append([0,1,0,0])
                elif (label == [1,0]).all():
                    Ymix.append([0,0,1,0])
                elif (label == [1,1]).all():
                    Ymix.append([0,0,0,1])    
            Y = jnp.array(Ymix)
            
            # The indexes of test are
            # All indexes (only analitical) \ train_index
            test_index = np.setdiff1d(np.arange(len(Y)), train_index)
            
            X_train, Y_train = X[train_index], Y[train_index]
            X_test, Y_test   = X[test_index], Y[test_index]
            
        if circuit:
            # Display the circuit
            print("+--- CIRCUIT ---+")
            print(self.drawer)

        @qml.qnode(self.device, interface="jax")
        def qcnn_circuit_prob(vqe_p, qcnn_p):
            self.vqe_qcnn_circuit(vqe_p, qcnn_p)

            return qml.probs([int(k) for k in self.final_active_wires])
        
        params = copy.copy(self.params)
        
        # Gradient of the Loss function
        jd_loss_fn = jax.jit(
            jax.grad(lambda p: loss_fn(X_train, Y_train, p, qcnn_circuit_prob))
        )

        # Update function
        # Returns updated parameters, updated state of the optimizer
        def update(params, opt_state):
            grads = jd_loss_fn(params)
            opt_state = opt_update(0, grads, opt_state)
            
            return get_params(opt_state), opt_state
        
        # Definying following function:
        # jitted loss function for training set loss(params)
        train_loss_fn = jax.jit(
            lambda p: loss_fn(X_train, Y_train, p, qcnn_circuit_prob)
        )
        # jitted loss function for test set loss(params)
        test_loss_fn = jax.jit(
            lambda p: loss_fn(X_test, Y_test, p, qcnn_circuit_prob)
        )

        # Initialize tqdm progress bar
        progress = tqdm.tqdm(range(n_epochs), position=0, leave=True)

        # Defining an optimizer in Jax
        opt_init, opt_update, get_params = optimizers.adam(lr)
        opt_state = opt_init(params)
        
        loss_history, loss_history_test = [], []
        # Training loop:
        for epoch in range(n_epochs):
            params, opt_state = update(params, opt_state)

            # Every 100 iterations append the updated training (and testing) loss
            if epoch % 100 == 0:
                loss_history.append(train_loss_fn(params))
                if len(Y_test) > 0:
                    loss_history_test.append(test_loss_fn(params))
                    
            # Update progress bar
            progress.update(1)
            progress.set_description("Cost: {0}".format(loss_history[-1]))

        # Update qcnn class after training
        self.loss_train = loss_history
        self.loss_test = loss_history_test
        self.params = params
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

    def save(self, filename):
        """
        Saves QCNN parameters to file

        Parameters
        ----------
        filename : str
            File where to save the parameters
        """

        things_to_save = [self.params, self.circuit_fun]

        with open(filename, "wb") as f:
            pickle.dump(things_to_save, f)


def load(filename_vqe, filename_qcnn):
    """
    Load QCNN from VQE file and QCNN file
    
    Parameters
    ----------
    filename_vqe : str
        Name of the file from where to load the VQE class
    filename_qcnn : str
        Name of the file from where to load the main parameters of the QCNN class
        
    Returns
    -------
    class
        QCNN class
    """
    loaded_vqe = vqe.load_vqe(filename_vqe)

    with open(filename_qcnn, "rb") as f:
        params, qcnn_circuit_fun = pickle.load(f)

    loaded_qcnn = qcnn(loaded_vqe, qcnn_circuit_fun)
    loaded_qcnn.params = params

    return loaded_qcnn

def get_trainset_gaussian(vqeclass, nS, sigma = 1):
    side = int(np.sqrt(vqeclass.n_states))
    if nS > 2*side - 1:
        return
    nS = nS // 2
    mu = side // 2
    
    training_set = []
    # Get Y training set:
    while len(training_set) < nS:
        sample = int(np.random.normal(mu, sigma))
        if sample not in training_set:
            if sample >= 0 and sample < side:
                training_set.append(sample)
    # Get X training set: 
    while len(training_set) < 2*nS:
        sample = int(np.random.normal(mu, sigma)) + side
        if sample not in training_set:
            if sample >= side and sample < 2*side:
                training_set.append(sample)
                
    return np.array(training_set)

def ANNNI_accuracy(qcnnclass, plot = False):
    circuit = qcnnclass.vqe_qcnn_circuit
    side = int(np.sqrt(qcnnclass.n_states))
    
    @qml.qnode(qcnnclass.device, interface="jax")
    def qcnn_circuit_prob(params_vqe, params):
        circuit(params_vqe, params)
    
        return [qml.probs(wires=int(k)) for k in qcnnclass.final_active_wires]
    
    vcircuit = jax.vmap(lambda v: qcnn_circuit_prob(v, qcnnclass.params), in_axes=(0))

    predictions = np.array(np.argmax(vcircuit(qcnnclass.vqe_params), axis = 2))

    labels       = []
    for idx in range(side*side):
        # compute coordinates and normalize for x in [0,1]
        # and y in [0,2]
        x = (idx//side)/side
        y = 2*(idx%side)/side
        
        if x == 0:
            if 1 <= y:
                labels.append([1,1])
            else:
                labels.append([0,1])
        elif x <= .5:
            if (qmlgen.paraferro(x) <= y):
                labels.append([1,1])
            else:
                labels.append([0,1])
        else:
            if (qmlgen.antiferro(x)) <= y:
                labels.append([1,1])
            else:
                labels.append([1,0])

    correct = (np.sum(np.array(labels) == predictions, axis = 1).astype(int) == 2)
    accuracy = np.sum(correct)/(side*side)

    if plot:
        plt.imshow(np.rot90(np.reshape(correct, (side,side))), cmap = 'RdYlGn')
        plt.show()
        
    return accuracy
