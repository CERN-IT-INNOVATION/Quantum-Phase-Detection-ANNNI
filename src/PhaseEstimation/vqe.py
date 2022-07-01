""" This module implements the base function to implement a VQE for a Ising Chain with Transverse Field. """
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from matplotlib import pyplot as plt

import copy
import tqdm  # Pretty progress bars
import joblib, pickle  # Writing and loading

import warnings

warnings.filterwarnings(
    "ignore",
    message="For Hamiltonians, the eigenvalues will be computed numerically. This may be computationally intensive for a large number of wires.Consider using a sparse representation of the Hamiltonian with qml.SparseHamiltonian.",
)

##############


def qml_build_H(N, lam, J):
    """
    Set up Hamiltonian:
            H = lam*Σsigma^i_z - J*Σsigma^i_x*sigma^{i+1}

    Parameters
    ----------
    N : int
        Number of spins of the Ising Chain
    lam : float
        Strenght of (transverse) magnetic field
    J : float
        Interaction strenght between spins

    Returns
    -------
    pennylane.ops.qubit.hamiltonian.Hamiltonian
        Hamiltonian Pennylane class for the (Transverse) Ising Chain
    """
    # Interaction of spins with magnetic field
    H = +lam * qml.PauliZ(0)
    for i in range(1, N):
        H = H + lam * qml.PauliZ(i)

    # Interaction between spins:
    for i in range(0, N - 1):
        H = H + J * (-1) * (qml.PauliX(i) @ qml.PauliX(i + 1))

    return H


def circuit_wall_RY(N, param, index=0):
    """
    Apply independent RY rotations to each wire in a Pennylane circuit

    Parameters
    ----------
    N : int
        Number of qubits
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    # Apply RY to each wire:
    for spin in range(N):
        qml.RY(param[index + spin], wires=spin)

    return index + N


def circuit_wall_RYRX(N, param, index=0):
    """
    Apply independent RX & RY rotations to each wire in a Pennylane circuit

    Parameters
    ----------
    N : int
        Number of qubits
    params: np.ndarray
        Array of parameters/rotation for the circuit
    index: int
        Index from where to pick the elements from the params array

    Returns
    -------
    int
        Updated starting index of params array for further rotations
    """
    # Apply RX and RY to each wire:
    for spin in range(N):
        qml.RY(param[index + spin], wires=spin)
        qml.RX(param[index + N + spin], wires=spin)

    return index + 2 * N


def circuit_wall_CNOT(N):
    """
    Apply CNOTs to every neighbouring qubits

    Parameters
    ----------
    N : int
        Number of qubits
    """
    # Apply entanglement to the neighbouring spins
    for spin in range(0, N - 1):
        qml.CNOT(wires=[spin, spin + 1])


def vqe_circuit(N, params):
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
    index = circuit_wall_RYRX(N, params)
    qml.Barrier()
    circuit_wall_CNOT(N)
    qml.Barrier()
    index = circuit_wall_RYRX(N, params, index)
    qml.Barrier()
    circuit_wall_CNOT(N)
    qml.Barrier()
    index = circuit_wall_RY(N, params, index)

    return index


class vqe:
    def __init__(self, N, J, l_steps, circuit):
        """
        Class for the VQE algorithm

        Parameters
        ----------
        N : int
            Number of spins in the Ising Chain / qubits of the circuit
        J : float
            Interaction strenght between spins
        l_steps : int
            Number of different states, namely defined by different magnetic field intensities = np.linspace(0, 2*J, l_steps)
        circuit : function
            Function of the VQE circuit
        """
        self.N = N
        self.J = J
        self.n_states = l_steps
        self.circuit = lambda p: circuit(self.N, p)
        self.n_params = self.circuit([0] * 10000)
        self.vqe_states = np.random.rand(self.n_states, self.n_params)
        self.device = qml.device("default.qubit.jax", wires=N, shots=None)

        self.lams = np.linspace(0, 2 * self.J, self.n_states)

        qml_Hs = []
        mat_Hs = []
        true_e = []
        labels = []

        for i, lam in enumerate(self.lams):
            labels.append(0) if lam <= J else labels.append(1)

            H = qml_build_H(self.N, float(lam), float(self.J))
            # Pennylane matrices
            qml_Hs.append(H)
            # True groundstate energies
            true_e.append(np.min(qml.eigvals(H)))
            # Standard matrix for of the hamiltonians
            mat_Hs.append(qml.matrix(H))

        self.true_e = jnp.array(true_e)
        self.qml_Hs = qml_Hs
        self.mat_Hs = jnp.array(mat_Hs)
        self.labels = labels
        self.MSE = []
        self.vqe_e = []
        self.recycle = False
        self.states_dist = []

        self.circuit_fun = circuit

    def show_circuit(self):
        """
        Prints the current circuit defined by self.circuit
        """

        @qml.qnode(self.device, interface="jax")
        def vqe_state(self):
            self.circuit(np.arange(self.n_params))

            return qml.state()

        drawer = qml.draw(vqe_state)
        print(drawer(self))

    def show_results(self):
        """
        Shows results of a trained VQE run:
        > VQE enegies plot
        > Loss curve if VQE was trained using recycle = False
        > Final relative errors
        > Mean Squared difference between final subsequent states
        """
        if len(self.MSE) > 0:
            tot_plots = 3 if self.recycle else 4
            fig, ax = plt.subplots(tot_plots, 1, figsize=(12, 18.6))

            ax[0].plot(self.lams, self.true_e, "--", label="True", color="red", lw=2)
            ax[0].plot(self.lams, self.vqe_e, ".", label="VQE", color="green", lw=2)
            ax[0].plot(self.lams, self.vqe_e, color="green", lw=2, alpha=0.6)
            ax[0].grid(True)
            ax[0].set_title(
                "Ground States of Ising Hamiltonian ({0}-spins), J = {1}".format(
                    self.N, self.J
                )
            )
            ax[0].set_xlabel(r"$\lambda$")
            ax[0].set_ylabel(r"$E(\lambda)$")
            ax[0].legend()

            k = 1
            if not self.recycle:
                ax[1].plot(
                    np.arange(len(self.MSE)) * 1000, self.MSE, ".", color="orange", ms=7
                )
                ax[1].plot(
                    np.arange(len(self.MSE)) * 1000, self.MSE, color="orange", alpha=0.4
                )
                ax[1].set_title("Convergence of VQE")
                ax[1].set_xlabel("Epoch")
                ax[1].set_ylabel("MSE")
                ax[1].grid(True)
                ax[1].axhline(y=0, color="r", linestyle="--")

                k = 2

            accuracy = np.abs((self.true_e - self.vqe_e) / self.true_e)
            ax[k].fill_between(
                self.lams, 0.01, max(np.max(accuracy), 0.01), color="r", alpha=0.3
            )
            ax[k].fill_between(
                self.lams, 0.01, min(np.min(accuracy), 0), color="green", alpha=0.3
            )
            ax[k].axhline(y=0.01, color="r", linestyle="--")
            ax[k].scatter(self.lams, accuracy)
            ax[k].grid(True)
            ax[k].set_title("Accuracy of VQE".format(self.N, self.J))
            ax[k].set_xlabel(r"$\lambda$")
            ax[k].set_ylabel(r"$|(E_{vqe} - E_{true})/E_{true}|$")

            ax[k + 1].set_title(
                "Mean square distance between consecutives density matrices"
            )
            ax[k + 1].plot(
                np.linspace(0, 2 * self.J, num=self.n_states - 1),
                self.states_dist,
                "-o",
            )
            ax[k + 1].grid(True)
            ax[k + 1].axvline(x=self.J, color="gray", linestyle="--")
            ax[k + 1].set_xlabel(r"$\lambda$")

            plt.tight_layout()

    def train(self, lr, n_epochs, reg=0, circuit=False, recycle=True):
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
            self.show_circuit()

        ### JAX FUNCTIONS ###
        # Circuit that returns the state |psi>
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

        # computes <psi|H|psi>
        def compute_E(state, Hmat):
            return jnp.conj(state) @ Hmat @ state

        # vmapped function for <psi|H|psi>
        v_compute_E = jax.vmap(compute_E, in_axes=(0, 0))
        # jitted function for <psi|H|psi>
        j_compute_E = jax.jit(compute_E)

        # Same function as above but returns the energies not MSE
        def compute_vqe_E(param, Hmat):
            pred_states = j_vqe_state(param)
            vqe_e = j_compute_E(pred_states, Hmat)

            return jnp.real(vqe_e)

        j_compute_vqe_E = jax.jit(compute_vqe_E)
        v_compute_vqe_E = jax.vmap(compute_vqe_E, in_axes=(0, 0))

        # Same function as above but returns the energies not MSE
        def v_compute_vqe_E(params):
            pred_states = v_vqe_state(params)
            vqe_e = v_compute_E(pred_states, self.mat_Hs)

            return jnp.real(vqe_e)

        j_v_compute_vqe_E = jax.jit(v_compute_vqe_E)

        # Prepare initial parameters randomly for each datapoint/state
        # We start from the same datapoint
        params = jnp.array(np.tile(np.random.rand(self.n_params), (self.n_states, 1)))

        if not recycle:
            self.recycle = False
            # Regularizator of the optimizer
            def compute_diff_states(states):
                return jnp.mean(jnp.square(jnp.diff(jnp.real(states), axis=1)))

            # Computes MSE of the true energies - vqe energies: function to minimize
            def update(params):
                pred_states = v_vqe_state(params)
                vqe_e = v_compute_E(pred_states, self.mat_Hs)

                if reg != 0:
                    return jnp.mean(jnp.real(vqe_e)) + reg * compute_diff_states(
                        pred_states
                    )
                else:
                    return jnp.mean(jnp.real(vqe_e))

            # Grad function of the MSE, used in updating the parameters
            jd_update = jax.jit(jax.grad(update))

            progress = tqdm.tqdm(range(n_epochs), position=0, leave=True)

            MSE = []
            for it in progress:
                params -= lr * jd_update(params)
                # I want to skip when it == 0
                if (it + 1) % 1000 == 0:
                    MSE.append(
                        jnp.mean(jnp.square(j_v_compute_vqe_E(params) - self.true_e))
                    )

                    # Update progress bar
                    progress.set_description("Cost: {0}".format(MSE[-1]))
        else:
            self.recycle = True
            # Computes MSE of the true energies - vqe energies: function to minimize
            def update(param, Hmat):
                pred_state = j_vqe_state(param)
                vqe_e = j_compute_E(pred_state, Hmat)

                return jnp.real(vqe_e)

            # Grad function of the MSE, used in updating the parameters
            jd_update = jax.jit(jax.grad(update))

            progress = tqdm.tqdm(enumerate(self.lams), position=0, leave=True)
            params = []
            param = jnp.array(np.random.rand(self.n_params))
            MSE = []
            for idx, lam in progress:
                MSE_idx = []
                epochs = n_epochs if idx > 0 else n_epochs * 10
                for it in range(epochs):
                    param -= lr * jd_update(param, self.mat_Hs[idx])

                    if (it + 1) % 100 == 0:
                        MSE_idx.append(
                            jnp.mean(
                                jnp.square(
                                    j_compute_vqe_E(param, self.mat_Hs[idx])
                                    - self.true_e[idx]
                                )
                            )
                        )
                params.append(copy.copy(param))
                progress.set_description("{0}/{1}".format(idx + 1, len(self.lams)))
                MSE.append(MSE_idx)
            MSE = np.mean(MSE, axis=0)
            params = jnp.array(params)

        self.MSE = MSE
        self.vqe_states = params
        self.vqe_e = j_v_compute_vqe_E(params)

        states = jv_vqe_state(params)
        self.states_dist = [
            np.mean(np.square(np.real(states[k + 1] - states[k])))
            for k in range(self.n_states - 1)
        ]

    def save(self, filename):
        """
        Save main parameters of the VQE class to a local file.
        Parameters saved:
        > N, J, number of states, parameters, circuit function

        Parameters
        ----------
        filename : str
            Local file to save the parameters
        """
        things_to_save = [
            self.N,
            self.J,
            self.n_states,
            self.vqe_states,
            self.circuit_fun,
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

    N, J, n_states, vqe_states, circuit_fun = things_to_load

    loaded_vqe = vqe(N, J, n_states, circuit_fun)
    loaded_vqe.vqe_states = vqe_states

    return loaded_vqe
