### IMPORTS ###
# Quantum libraries:
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

# Plotting
from matplotlib import pyplot as plt

# Other
import copy
import tqdm  # Pretty progress bars
import joblib  # Writing and loading
from noisyopt import minimizeSPSA

import multiprocessing

##############

#   _
#  / |
#  | |
#  | |_
#  |_(_) General


def qml_build_H(N, lam, J):
    """
    Set up Hamiltonian:
            H = lam*Σsigma^i_z - J*Σsigma^i_x*sigma^{i+1}_
    """
    # Interaction of spins with magnetic field
    H = +lam * qml.PauliZ(0)
    for i in range(1, N):
        H = H + lam * qml.PauliZ(i)

    # Interaction between spins:
    for i in range(0, N - 1):
        H = H + J * (-1) * (qml.PauliX(i) @ qml.PauliX(i + 1))

    return H


#  ____
# |___ \
#   __) |
#  / __/ _
# |_____(_) Circuit functions


def circuit_wall_RY(N, param, index = 0):
    # Apply RX and RY to each wire:
    for spin in range(N):
        qml.RY(param[index + spin], wires=spin)

    return index + N


def circuit_wall_RYRX(N, param, index = 0):
    # Apply RX and RY to each wire:
    for spin in range(N):
        qml.RY(param[index + spin], wires=spin)
        qml.RX(param[index + N + spin], wires=spin)

    return index + 2 * N

def circuit_wall_CNOT(N, params):
    # Apply entanglement to the neighbouring spins
    for spin in range(0, N - 1):
        qml.CNOT(wires=[spin, spin + 1])

def vqe_circuit(N, params):
    index = circuit_wall_RYRX(N, params)
    qml.Barrier()
    circuit_wall_CNOT(N, params)
    qml.Barrier()
    index = circuit_wall_RYRX(N, params, index)
    qml.Barrier()
    circuit_wall_CNOT(N, params)
    qml.Barrier()
    index = circuit_wall_RY(N, params, index)
    
    return index


#  _____
# |___ /
#   |_ \
#  ___) |
# |____(_) Learning functions


def vqe_train_jax(
    step_size,
    n_epochs,
    N,
    J,
    l_steps,
    device,
    vqe_circuit_fun,
    reg=0,
    circuit=False,
    plots=False,
    parameter_info=True,
):
    # IDEA: Making the Jax version of a VQE eigensolver is a bit less intuitive that in the QCNN learning function,
    #       since here we have l_steps different circuit since each output of the circuit is <psi|H|psi> where H,
    #       changes for each datapoints.
    #       Here the output of each circuit is |psi>, while the Hs is moved to the loss function
    #       <PSI|Hs|PSI> is computed through 2 jax.einsum

    # circuit functions returns the number of parameters needed for the circuit itself
    n_params = vqe_circuit_fun(N, [0] * 1000)

    if parameter_info:
        print("+--- PARAMETERS ---+")
        print("step_size      = {0} (Step size of the optimizer)".format(step_size))
        print("n_epochs       = {0} (# epochs for the other GSs)".format(n_epochs))
        print("N              = {0} (Number of spins of the system)".format(N))

    ### JAX FUNCTIONS ###

    # Circuit that returns the state |psi>
    @qml.qnode(device, interface="jax")
    def vqe_state(vqe_params, N):
        vqe_circuit_fun(N, vqe_params)

        return qml.state()

    if circuit:
        # Display the circuit
        print("+--- CIRCUIT ---+")
        drawer = qml.draw(vqe_state)
        print(drawer(np.arange(n_params), N))

    # vmap of the circuit
    v_vqe_state = jax.vmap(lambda v: vqe_state(v, N), in_axes=(0))

    # jitted vmap of the circuit
    jv_vqe_state = jax.jit(v_vqe_state)

    # computes <psi|H|psi>
    def compute_E(state, Hmat):
        return jnp.conj(state) @ Hmat @ state

    # vmapped function for <psi|H|psi>
    v_compute_E = jax.vmap(compute_E, in_axes=(0, 0))

    # Regularizator of the optimizer
    def compute_diff_states(states):
        return jnp.mean(jnp.square(jnp.diff(jnp.real(states), axis=1)))

    # Computes MSE of the true energies - vqe energies: function to minimize
    def update(params):
        pred_states = v_vqe_state(params)
        vqe_e = v_compute_E(pred_states, Hsmat)

        if reg != 0:
            return jnp.mean(
                jnp.square(jnp.real(vqe_e) - true_e)
            ) + reg * compute_diff_states(pred_states)
        else:
            return jnp.mean(jnp.square(jnp.real(vqe_e) - true_e))

    # Same function as above but returns the energies not MSE
    def compute_vqe(params):
        pred_states = v_vqe_state(params)
        vqe_e = v_compute_E(pred_states, Hsmat)

        return jnp.real(vqe_e)

    # Grad function of the MSE, used in updating the parameters
    jd_update = jax.jit(jax.grad(update))

    j_compute_vqe = jax.jit(compute_vqe)

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
    energy_err = [0] * (len(lams))
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

    progress = tqdm.tqdm(range(n_epochs), position=0, leave=True)

    MSE = []
    for it in progress:
        params -= step_size * jd_update(params)
        # I want to skip when it == 0
        if (it + 1) % 1000 == 0:
            MSE.append(jnp.mean(jnp.square(j_compute_vqe(params) - true_e)))

            # Update progress bar
            progress.set_description("Cost: {0}".format(MSE[-1]))

    vqe_e = j_compute_vqe(params)

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

        ax[1].plot(np.arange(len(MSE)) * 1000, MSE, ".", color="orange", ms=7)
        ax[1].plot(np.arange(len(MSE)) * 1000, MSE, color="orange", alpha=0.4)
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


def vqe_train_jax_recycled(
    step_size,
    n_epochs,
    N,
    J,
    l_steps,
    device,
    vqe_circuit_fun,
    reg=0,
    circuit=False,
    plots=False,
    parameter_info=True,
):
    # IDEA: Making the Jax version of a VQE eigensolver is a bit less intuitive that in the QCNN learning function,
    #       since here we have l_steps different circuit since each output of the circuit is <psi|H|psi> where H,
    #       changes for each datapoints.
    #       Here the output of each circuit is |psi>, while the Hs is moved to the loss function
    #       <PSI|Hs|PSI> is computed through 2 jax.einsum

    # circuit functions returns the number of parameters needed for the circuit itself
    n_params = vqe_circuit_fun(N, [0] * 1000)

    if parameter_info:
        print("+--- PARAMETERS ---+")
        print("step_size      = {0} (Step size of the optimizer)".format(step_size))
        print("n_epochs       = {0} (# epochs for the other GSs)".format(n_epochs))
        print("N              = {0} (Number of spins of the system)".format(N))

    ### JAX FUNCTIONS ###

    # Circuit that returns the state |psi>
    @qml.qnode(device, interface="jax")
    def vqe_state(vqe_params, N):
        vqe_circuit_fun(N, vqe_params)

        return qml.state()

    if circuit:
        # Display the circuit
        print("+--- CIRCUIT ---+")
        drawer = qml.draw(vqe_state)
        print(drawer(np.arange(n_params), N))

    j_vqe_state = jax.jit(lambda p: vqe_state(p, N))
    v_vqe_state = jax.vmap(lambda p: vqe_state(p, N))

    # computes <psi|H|psi>
    def compute_E(state, Hmat):
        return jnp.conj(state) @ Hmat @ state

    # vmapped function for <psi|H|psi>
    v_compute_E = jax.vmap(compute_E, in_axes=(0, 0))
    j_compute_E = jax.jit(compute_E)

    # Computes MSE of the true energies - vqe energies: function to minimize
    def update(param, Hmat):
        pred_state = j_vqe_state(param)
        vqe_e = j_compute_E(pred_state, Hmat)

        return jnp.real(vqe_e)

    # Same function as above but returns the energies not MSE
    def compute_vqe(param, Hmat):
        pred_states = j_vqe_state(param)
        vqe_e = j_compute_E(pred_states, Hmat)

        return jnp.real(vqe_e)

    # Grad function of the MSE, used in updating the parameters
    jd_update = jax.jit(jax.grad(update))
    j_compute_vqe = jax.jit(compute_vqe)
    v_compute_vqe = jax.vmap(compute_vqe, in_axes=(0, 0))

    lams = np.linspace(0, 2 * J, l_steps)

    # For each lamda create optimizer and H
    Hs = []
    Hsmat = []
    opts = []
    energy_err = [0] * (len(lams))
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

    progress = tqdm.tqdm(enumerate(lams), position=0, leave=True)

    params = []

    param = jnp.array(np.random.rand(n_params))
    for idx, lam in progress:
        for it in range(n_epochs):
            param -= step_size * jd_update(param, Hsmat[idx])

        params.append(copy.copy(param))
        progress.set_description("{0}/{1}".format(idx, len(lams)))

    params = jnp.array(params)
    vqe_e = v_compute_vqe(params, Hsmat)

    if plots:
        fig, ax = plt.subplots(4, 1, figsize=(12, 14))

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
        ax[1].fill_between(
            lams, 0.01, max(np.max(accuracy), 0.01), color="r", alpha=0.3
        )
        ax[1].fill_between(
            lams, 0.01, min(np.min(accuracy), 0), color="green", alpha=0.3
        )
        ax[1].axhline(y=0.01, color="r", linestyle="--")
        ax[1].scatter(lams, accuracy)
        ax[1].grid(True)
        ax[1].set_title("Accuracy of VQE".format(N, J))
        ax[1].set_xlabel(r"$\lambda$")
        ax[1].set_ylabel(r"$|(E_{vqe} - E_{true})/E_{true}|$")

        states = v_vqe_state(params)
        rho_dist = [
            np.mean(np.square(np.real(states[k + 1] - states[k])))
            for k in range(l_steps - 1)
        ]

        ax[2].set_title("Mean square distance between consecutives density matrices")
        ax[2].plot(np.linspace(0, 2 * J, num=l_steps - 1), rho_dist, "-o")
        ax[2].grid(True)
        ax[2].axvline(x=J, color="gray", linestyle="--")
        ax[2].set_xlabel(r"$\lambda$")

        plt.tight_layout()

    ys = []
    for l in lams:
        ys.append(0) if l <= J else ys.append(1)

    return vqe_e, params, ys


#  _  _
# | || |
# | || |_
# |__   _|
#    |_|(_) Visualization
def show_train_plots(data, N, J, device, vqe_circuit_fun, lsteps=100):
    @qml.qnode(device)
    def vqe_cost_fn(vqe_params, N, H, p_noise=0, p_noise_ent=0):
        vqe_circuit_fun(N, vqe_params, p_noise, p_noise_ent)

        # return <psi|H|psi>
        return qml.expval(H)

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
