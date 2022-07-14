import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import plotly.graph_objects as go
import pandas as pd
from orqviz.scans import perform_2D_scan, plot_2D_scan_result
from orqviz.pca import (get_pca, perform_2D_pca_scan, plot_pca_landscape, 
                        plot_optimization_trajectory_on_pca)

def show_VQE_isingchain(vqeclass):
    """
    Shows results of a trained VQE run:
    > VQE enegies plot
    > Loss curve if VQE was trained using recycle = False
    > Final relative errors
    > Mean Squared difference between final subsequent states
    """
    vqeclass.states_dist = [
        np.mean(np.square(np.real(vqeclass.states[k + 1] - vqeclass.states[k])))
        for k in range(vqeclass.n_states - 1)
    ]

    lams = np.linspace(0, 2*vqeclass.Hs.J, vqeclass.n_states)

    tot_plots = 3 if vqeclass.recycle else 4
    fig, ax = plt.subplots(tot_plots, 1, figsize=(12, 18.6))

    ax[0].plot(lams, vqeclass.Hs.true_e, "--", label="True", color="red", lw=2)
    ax[0].plot(lams, vqeclass.vqe_e, ".", label="VQE", color="green", lw=2)
    ax[0].plot(lams, vqeclass.vqe_e, color="green", lw=2, alpha=0.6)
    ax[0].grid(True)
    ax[0].set_title(
        "Ground States of Ising Hamiltonian ({0}-spins), J = {1}".format(
            vqeclass.N, vqeclass.Hs.J
        )
    )
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(r"$E(\lambda)$")
    ax[0].legend()

    k = 1
    if not vqeclass.recycle:
        ax[1].plot(
            np.arange(len(vqeclass.MSE)) * 100, vqeclass.MSE, ".", color="orange", ms=7
        )
        ax[1].plot(
            np.arange(len(vqeclass.MSE)) * 100, vqeclass.MSE, color="orange", alpha=0.4
        )
        ax[1].set_title("Convergence of VQE")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("MSE")
        ax[1].grid(True)
        ax[1].axhline(y=0, color="r", linestyle="--")

        k = 2

    accuracy = np.abs((vqeclass.Hs.true_e - vqeclass.vqe_e) / vqeclass.Hs.true_e)
    ax[k].fill_between(
        lams, 0.01, max(np.max(accuracy), 0.01), color="r", alpha=0.3
    )
    ax[k].fill_between(
        lams, 0.01, min(np.min(accuracy), 0), color="green", alpha=0.3
    )
    ax[k].axhline(y=0.01, color="r", linestyle="--")
    ax[k].scatter(lams, accuracy)
    ax[k].grid(True)
    ax[k].set_title("Accuracy of VQE".format(vqeclass.N, vqeclass.Hs.J))
    ax[k].set_xlabel(r"$\lambda$")
    ax[k].set_ylabel(r"$|(E_{vqe} - E_{true})/E_{true}|$")

    ax[k + 1].set_title(
        "Mean square distance between consecutives density matrices"
    )
    ax[k + 1].plot(
        np.linspace(0, 2 * vqeclass.Hs.J, num=vqeclass.n_states - 1),
        vqeclass.states_dist,
        "-o",
    )
    ax[k + 1].grid(True)
    ax[k + 1].axvline(x=vqeclass.Hs.J, color="gray", linestyle="--")
    ax[k + 1].set_xlabel(r"$\lambda$")

    plt.tight_layout()
    
def show_VQE_nnisingchain(vqeclass):
    """
    Shows results of a trained VQE run:
    > VQE enegies plot
    > Loss curve if VQE was trained using recycle = False
    > Final relative errors
    > Mean Squared difference between final subsequent states
    """
    vqeclass.states_dist = [
        np.mean(np.square(np.real(vqeclass.states[k + 1] - vqeclass.states[k])))
        for k in range(vqeclass.n_states - 1)
    ]

    j2s = np.linspace(0, 1*vqeclass.Hs.J1, vqeclass.n_states)

    tot_plots = 3 if vqeclass.recycle else 4
    fig, ax = plt.subplots(tot_plots, 1, figsize=(12, 18.6))

    ax[0].plot(j2s, vqeclass.Hs.true_e, "--", label="True", color="red", lw=2)
    ax[0].plot(j2s, vqeclass.vqe_e, ".", label="VQE", color="green", lw=2)
    ax[0].plot(j2s, vqeclass.vqe_e, color="green", lw=2, alpha=0.6)
    ax[0].grid(True)
    ax[0].set_title(
        "Ground States of Ising Hamiltonian ({0}-spins), J1 = {1}".format(
            vqeclass.N, vqeclass.Hs.J1
        )
    )
    ax[0].set_xlabel(r"$J1/J2$")
    ax[0].set_ylabel(r"$E(\lambda)$")
    ax[0].legend()

    k = 1
    if not vqeclass.recycle:
        ax[1].plot(
            np.arange(len(vqeclass.MSE)) * 100, vqeclass.MSE, ".", color="orange", ms=7
        )
        ax[1].plot(
            np.arange(len(vqeclass.MSE)) * 100, vqeclass.MSE, color="orange", alpha=0.4
        )
        ax[1].set_title("Convergence of VQE")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("MSE")
        ax[1].grid(True)
        ax[1].axhline(y=0, color="r", linestyle="--")

        k = 2

    accuracy = np.abs((vqeclass.Hs.true_e - vqeclass.vqe_e) / vqeclass.Hs.true_e)
    ax[k].fill_between(
        j2s, 0.01, max(np.max(accuracy), 0.01), color="r", alpha=0.3
    )
    ax[k].fill_between(
        j2s, 0.01, min(np.min(accuracy), 0), color="green", alpha=0.3
    )
    ax[k].axhline(y=0.01, color="r", linestyle="--")
    ax[k].scatter(j2s, accuracy)
    ax[k].grid(True)
    ax[k].set_title("Accuracy of VQE")
    ax[k].set_xlabel(r"$\lambda$")
    ax[k].set_ylabel(r"$|(E_{vqe} - E_{true})/E_{true}|$")

    ax[k + 1].set_title(
        "Mean square distance between consecutives density matrices"
    )
    ax[k + 1].plot(
        np.linspace(0, 1 * vqeclass.Hs.J1, num=vqeclass.n_states - 1),
        vqeclass.states_dist,
        "-o",
    )
    ax[k + 1].grid(True)
    ax[k + 1].axvline(x=vqeclass.Hs.J1, color="gray", linestyle="--")
    ax[k + 1].set_xlabel(r"$\lambda$")

    plt.tight_layout()

def show_VQE_annni(vqeclass, log_heatmap = False):
    """
    Shows results of a trained VQE run:
    > VQE enegies plot
    > Loss curve if VQE was trained using recycle = False
    > Final relative errors
    > Mean Squared difference between final neighbouring states
    """
    states_dist = []
    side = int(np.sqrt(vqeclass.n_states))

    trues = np.reshape(vqeclass.Hs.true_e,(side, side) )
    preds = np.reshape(vqeclass.vqe_e,(side, side) )

    x = np.linspace(1, 0, side)
    y = np.linspace(0, 2, side)

    fig = go.Figure(data=[go.Surface(opacity=.2, colorscale='Reds', z=trues, x=x, y=y),
                  go.Surface(opacity=1, colorscale='Blues',z=preds, x=x, y=y)])

    fig.update_layout(height=500)
    fig.show()

    if not vqeclass.recycle:
        plt.figure(figsize=(15,3))
        plt.title('Loss of training set')
        plt.plot(np.arange(len(vqeclass.MSE)+1)[1:]*100, vqeclass.MSE)
        plt.show()

    accuracy = np.rot90( np.abs(preds-trues)/np.abs(trues) )

    fig2, ax = plt.subplots(1, 2, figsize=(10, 40))

    if not log_heatmap:
        colors_good = np.squeeze( np.dstack((np.dstack((np.linspace(.3,0,25), np.linspace(.8,1,25))), np.linspace(1,0,25) )) )
        colors_bad  = np.squeeze( np.dstack((np.dstack((np.linspace(1,0,100), [0]*100)), [0]*100 )) )
        colors = np.vstack((colors_good, colors_bad))
        cmap_acc = LinearSegmentedColormap.from_list('accuracies', colors)

        acc = ax[0].imshow(accuracy, cmap = cmap_acc)
        acc.set_clim(0,0.05)
        plt.colorbar(acc, ax=ax[0], fraction=0.04)
    else:
        colors = np.squeeze( np.dstack((np.dstack((np.linspace(0,1,75), np.linspace(1,0,75))), np.linspace(0,0,75) )) )
        cmap_acc = LinearSegmentedColormap.from_list('accuracies', colors)
        acc = ax[0].imshow(accuracy, cmap = cmap_acc, norm=LogNorm())
        plt.colorbar(acc, ax=ax[0], fraction=0.04)

    ax[0].set_xlabel('L')
    ax[0].set_ylabel('K')
    ax[0].set_title('Relative errors')

    ax[0].set_xticks(ticks=np.linspace(0,side-1,4).astype(int), labels= np.round(x[np.linspace(side-1,0,4).astype(int)],2))
    ax[0].set_yticks(ticks=np.linspace(0,side-1,4).astype(int), labels= np.round(y[np.linspace(side-1,0,4).astype(int)],2))

    for idx, state in enumerate(vqeclass.states):
        neighbours = np.array([idx + 1, idx - 1, idx + side, idx - side])
        neighbours = np.delete(neighbours, np.logical_not(np.isin(neighbours, vqeclass.Hs.recycle_rule)) )


        if (idx + 1) % side == 0 and idx != vqeclass.n_states - 1:
            neighbours = np.delete(neighbours, 0)
        if (idx    ) % side == 0 and idx != 0:
            neighbours = np.delete(neighbours, 1)

        states_dist.append(np.mean(np.square([np.real(vqeclass.states[n] - state) for n in neighbours]) ) )

    ax[1].set_title('Mean square difference between neighbouring states')
    diff = ax[1].imshow(np.rot90(np.reshape(states_dist, (side,side)) ) )
    plt.colorbar(diff, ax=ax[1], fraction=0.04)
    ax[1].set_xlabel('L')
    ax[1].set_ylabel('K')

    ax[1].set_xticks(ticks=np.linspace(0,side-1,4).astype(int), labels= np.round(x[np.linspace(side-1,0,4).astype(int)],2))
    ax[1].set_yticks(ticks=np.linspace(0,side-1,4).astype(int), labels= np.round(y[np.linspace(side-1,0,4).astype(int)],2))
    plt.tight_layout()

def show_VQE_trajectory(vqeclass, idx):
    def loss(params):
        @qml.qnode(vqeclass.device, interface="jax")
        def vqe_state(vqe_params):
            vqeclass.circuit(vqe_params)

            return qml.state()

        pred_state = vqe_state(params)
        vqe_e = jnp.conj(pred_state) @ vqeclass.Hs.mat_Hs[idx] @ pred_state

        return jnp.real(vqe_e)

    trajs = []
    for traj in vqeclass.trajectory:
        trajs.append(traj[idx])

    dir1 = np.array([1., 0.])
    dir2 = np.array([0., 1.])

    pca = get_pca(trajs)
    scan_pca_result = perform_2D_pca_scan(pca, loss, n_steps_x=30)

    fig, ax = plt.subplots()
    plot_pca_landscape(scan_pca_result, pca, fig=fig, ax=ax)
    plot_optimization_trajectory_on_pca(trajs, pca, ax=ax, 
                                        label="Optimization Trajectory", color="lightsteelblue")
    plt.legend()
    plt.show()
    
def show_QCNN_classification1D(qcnnclass):
    """
    Plots performance of the classifier on the whole data
    """
    train_index = qcnnclass.train_index

    @qml.qnode(qcnnclass.device, interface="jax")
    def qcnn_circuit_prob(params_vqe, params):
        qcnnclass.circuit(params_vqe, params)

        return qml.probs(wires=qcnnclass.N - 1)

    test_index = np.setdiff1d(np.arange(len(qcnnclass.vqe_states)), train_index)

    predictions_train = []
    predictions_test = []

    colors_train = []
    colors_test = []

    vcircuit = jax.vmap(lambda v: qcnn_circuit_prob(v, qcnnclass.params), in_axes=(0))
    predictions = vcircuit(qcnnclass.vqe_states)[:, 1]

    for i, prediction in enumerate(predictions):
        # if data in training set
        if i in train_index:
            predictions_train.append(prediction)
            if np.round(prediction) == 0:
                colors_train.append("green") if qcnnclass.labels[
                    i
                ] == 0 else colors_train.append("red")
            else:
                colors_train.append("red") if qcnnclass.labels[
                    i
                ] == 0 else colors_train.append("green")
        else:
            predictions_test.append(prediction)
            if np.round(prediction) == 0:
                colors_test.append("green") if qcnnclass.labels[
                    i
                ] == 0 else colors_test.append("red")
            else:
                colors_test.append("red") if qcnnclass.labels[
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
    #ax[0].set_xlabel("Transverse field")
    ax[0].set_ylabel("Prediction of label II")
    ax[0].set_title("Predictions of labels; J = 1")
    ax[0].scatter(
        2 * np.sort(train_index) / len(qcnnclass.vqe_states),
        predictions_train,
        c="royalblue",
        label="Training samples",
    )
    ax[0].scatter(
        2 * np.sort(test_index) / len(qcnnclass.vqe_states),
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
    #ax[1].set_xlabel("Transverse field")
    ax[1].set_ylabel("Prediction of label II")
    ax[1].set_title("Predictions of labels; J = 1")
    ax[1].scatter(
        2 * np.sort(train_index) / len(qcnnclass.vqe_states),
        predictions_train,
        c=colors_train,
    )
    ax[1].scatter(
        2 * np.sort(test_index) / len(qcnnclass.vqe_states),
        predictions_test,
        c=colors_test,
    )