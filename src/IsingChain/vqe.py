""" This module implements the base function to implement a VQE for a Ising Chain with Transverse Field. """
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from matplotlib import pyplot as plt

import copy
import tqdm  # Pretty progress bars
import joblib  # Writing and loading
from noisyopt import minimizeSPSA

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


def vqe_train(
    lr,
    n_epochs,
    N,
    J,
    l_steps,
    vqe_circuit_fun,
    reg=0,
    circuit=False,
    plots=False,
    recycle=True,
):
    """
    Training function for the VQE.

    Parameters
    ----------
    lr : float
        Learning rate to be multiplied in the circuit-gradient output
    n_epochs : int
        Total number of epochs for each learning
    N : int
        Number of spins/qubits
    J : float
        Interaction strenght between spins
    l_steps : int
        Number of different magnetic field strenghts to train. (Taken equidistantly
        from 0 to 2*J
    vqe_circuit_fun : function
        Circuit function of the VQE
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

    Returns
    -------
    np.ndarray
        Array of energies of VQE states
    np.ndarray
        Array of parameters of the circuit for each state
    np.ndarray
        True labels of each state
    """
    # IDEA: Making the Jax version of a VQE eigensolver is a bit less intuitive that in the QCNN learning function,
    #       since here we have l_steps different circuits since each output of the circuit is <psi|H|psi> where H,
    #       changes for each datapoint.
    #       Here the output of each circuit is |psi>, while the Hs is moved to the loss function
    #       <PSI|Hs|PSI> is computed through two jax.einsum

    # circuit functions returns the number of parameters needed for the circuit itself
    n_params = vqe_circuit_fun(N, [0] * 1000)

    ### JAX FUNCTIONS ###

    device = qml.device("default.qubit.jax", wires=N, shots=None)

    # Circuit that returns the state |psi>
    @qml.qnode(device, interface="jax")
    def vqe_state(vqe_params, N):
        vqe_circuit_fun(N, vqe_params)

        return qml.state()

    # vmap of the circuit
    v_vqe_state = jax.vmap(lambda v: vqe_state(v, N), in_axes=(0))

    # jitted vmap of the circuit
    jv_vqe_state = jax.jit(v_vqe_state)

    # jitted circuit
    j_vqe_state = jax.jit(lambda p: vqe_state(p, N))

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
        vqe_e = v_compute_E(pred_states, Hsmat)

        return jnp.real(vqe_e)

    j_v_compute_vqe_E = jax.jit(v_compute_vqe_E)

    if circuit:
        # Display the circuit
        print("+--- CIRCUIT ---+")
        drawer = qml.draw(vqe_state)
        print(drawer(np.arange(n_params), N))

    lams = np.linspace(0, 2 * J, l_steps)

    # Prepare initial parameters randomly for each datapoint/state
    # We start from the same datapoint
    param = np.random.rand(n_params)
    params0 = []
    for _ in lams:
        params0.append(param)

    params = jnp.array(params0)

    # For each lamda create optimizer and H
    Hs = []
    Hsmat = []
    opts = []
    true_e = []

    for i, lam in enumerate(lams):
        # Pennylane matrices
        Hs.append(qml_build_H(N, float(lam), float(J)))
        # True groundstate energies
        true_e.append(np.min(qml.eigvals(Hs[i])))
        # Standard matrix for of the hamiltonians
        Hsmat.append(qml.matrix(Hs[-1]))

    true_e = jnp.array(true_e)
    Hsmat = jnp.array(Hsmat)

    if not recycle:
        # Regularizator of the optimizer
        def compute_diff_states(states):
            return jnp.mean(jnp.square(jnp.diff(jnp.real(states), axis=1)))

        # Computes MSE of the true energies - vqe energies: function to minimize
        def update(params):
            pred_states = v_vqe_state(params)
            vqe_e = v_compute_E(pred_states, Hsmat)

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
                MSE.append(jnp.mean(jnp.square(j_v_compute_vqe_E(params) - true_e)))

                # Update progress bar
                progress.set_description("Cost: {0}".format(MSE[-1]))
    else:
        # Computes MSE of the true energies - vqe energies: function to minimize
        def update(param, Hmat):
            pred_state = j_vqe_state(param)
            vqe_e = j_compute_E(pred_state, Hmat)

            return jnp.real(vqe_e)

        # Grad function of the MSE, used in updating the parameters
        jd_update = jax.jit(jax.grad(update))

        progress = tqdm.tqdm(enumerate(lams), position=0, leave=True)
        params = []
        param = jnp.array(np.random.rand(n_params))
        MSE = []
        for idx, lam in progress:
            MSE_idx = []
            epochs = n_epochs if idx > 0 else n_epochs * 10
            for it in range(epochs):
                param -= lr * jd_update(param, Hsmat[idx])

                if (it + 1) % 100 == 0:
                    MSE_idx.append(
                        jnp.mean(
                            jnp.square(j_compute_vqe_E(param, Hsmat[idx]) - true_e[idx])
                        )
                    )
            params.append(copy.copy(param))
            progress.set_description("{0}/{1}".format(idx, len(lams)))
            MSE.append(MSE_idx)
        MSE = np.mean(MSE, axis=0)
        params = jnp.array(params)

    vqe_e = j_v_compute_vqe_E(params)

    if plots:
        fig, ax = plt.subplots(4, 1, figsize=(12, 18.6))

        ax[0].plot(lams, true_e, "--", label="True", color="red", lw=2)
        ax[0].plot(lams, vqe_e, ".", label="VQE", color="green", lw=2)
        ax[0].plot(lams, vqe_e, color="green", lw=2, alpha=0.6)
        ax[0].grid(True)
        ax[0].set_title(
            "Ground States of Ising Hamiltonian ({0}-spins), J = {1}".format(N, J)
        )
        ax[0].set_xlabel(r"$\lambda$")
        ax[0].set_ylabel(r"$E(\lambda)$")
        ax[0].legend()

        mse_steps = 100 if recycle else 1000
        ax[1].plot(np.arange(len(MSE)) * mse_steps, MSE, ".", color="orange", ms=7)
        ax[1].plot(np.arange(len(MSE)) * mse_steps, MSE, color="orange", alpha=0.4)
        ax[1].set_title("Convergence of VQE")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("MSE")
        ax[1].grid(True)
        ax[1].axhline(y=0, color="r", linestyle="--")

        true_e = np.array(true_e)
        vqe_e = np.array(vqe_e)
        accuracy = np.abs((true_e - vqe_e) / true_e)
        ax[2].fill_between(
            lams, 0.01, max(np.max(accuracy), 0.01), color="r", alpha=0.3
        )
        ax[2].fill_between(
            lams, 0.01, min(np.min(accuracy), 0), color="green", alpha=0.3
        )
        ax[2].axhline(y=0.01, color="r", linestyle="--")
        ax[2].scatter(lams, accuracy)
        ax[2].grid(True)
        ax[2].set_title("Accuracy of VQE".format(N, J))
        ax[2].set_xlabel(r"$\lambda$")
        ax[2].set_ylabel(r"$|(E_{vqe} - E_{true})/E_{true}|$")

        states = jv_vqe_state(params)
        rho_dist = [
            np.mean(np.square(np.real(states[k + 1] - states[k])))
            for k in range(l_steps - 1)
        ]

        ax[3].set_title("Mean square distance between consecutives density matrices")
        ax[3].plot(np.linspace(0, 2 * J, num=l_steps - 1), rho_dist, "-o")
        ax[3].grid(True)
        ax[3].axvline(x=J, color="gray", linestyle="--")
        ax[3].set_xlabel(r"$\lambda$")

        plt.tight_layout()

    ys = []
    for l in lams:
        ys.append(0) if l <= J else ys.append(1)

    return vqe_e, params, ys


def show_train_plots(data, N, J, vqe_circuit_fun):
    """
    Display the results/quality of the VQE states

    Parameters
    ----------
    data : np.ndarray
        Array of lists (VQE_parameters, label)
    N : int
        Number of spins/qubits
    J : float
        Interaction strenght between spins
    vqe_circuit_fun : function
        Circuit function of the VQE
    """

    device = qml.device("default.qubit.jax", wires=N, shots=None)

    @qml.qnode(device)
    def vqe_cost_fn(vqe_params, N, H, p_noise=0, p_noise_ent=0):
        vqe_circuit_fun(N, vqe_params, p_noise, p_noise_ent)

        # return <psi|H|psi>
        return qml.expval(H)

    lsteps = len(data)
    lams = np.linspace(0, 2 * J, lsteps)

    true_e = []
    vqe_e = []
    for i, l in enumerate(lams):
        circ_params = data[i][0]

        H = qml_build_H(N, float(l), float(J))
        true_e.append(np.min(qml.eigvals(H)))
        vqe_e.append(vqe_cost_fn(circ_params, N, H))

    fig, ax = plt.subplots(2, 1, figsize=(12, 9.3))

    ax[0].plot(lams, true_e, "--", label="True", color="red", lw=2)
    ax[0].plot(lams, vqe_e, ".", label="VQE", color="green", lw=2)
    ax[0].plot(lams, vqe_e, color="green", lw=2, alpha=0.6)
    ax[0].grid(True)
    ax[0].set_title(
        "Ground States of Ising Hamiltonian ({0}-spins), J = {1}".format(N, J)
    )
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(r"$E(\lambda)$")
    ax[0].legend()

    true_e = np.array(true_e)
    vqe_e = np.array(vqe_e)
    accuracy = np.abs((true_e - vqe_e) / true_e)
    ax[1].fill_between(lams, 0.01, max(np.max(accuracy), 0.01), color="r", alpha=0.3)
    ax[1].fill_between(lams, 0.01, min(np.min(accuracy), 0), color="green", alpha=0.3)
    ax[1].axhline(y=0.01, color="r", linestyle="--")
    ax[1].scatter(lams, accuracy)
    ax[1].grid(True)
    ax[1].set_title("Accuracy of VQE".format(N, J))
    ax[1].set_xlabel(r"$\lambda$")
    ax[1].set_ylabel(r"$|(E_{vqe} - E_{true})/E_{true}|$")

    plt.tight_layout()
