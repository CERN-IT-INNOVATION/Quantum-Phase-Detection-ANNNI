import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import plotly.graph_objects as go

from PhaseEstimation import general as qmlgen
from PhaseEstimation import vqe

from typing import List, Callable

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
rc("text", usetex=True)


def getlines(
    func: Callable, xrange: List[float], side: int, color: str, res: int = 100
):
    """
    Plot function func from xrange[0] to xrange[1]
    """
    xs = np.linspace(xrange[0], xrange[1], res)
    ys = func(xs)
    plt.plot(side * xs - 0.5, side - ys * side / 2 - 0.5, color=color, alpha=0.8)


def show_VQE_isingchain(vqeclass: vqe.vqe, excited: bool = False):
    """
    Shows results of a trained VQE run

    Parameters
    ----------
    vqeclass : vqe.vqe
        Custom VQE class after being trained
    excited : bool
        if True -> tries to display the Excited states aswell
    """
    # Exit if the VQE was not trained for excited states
    if excited:
        try:
            vqeclass.vqe_params1
        except:
            return

    if not excited:
        true_e = vqeclass.true_e0
        vqe_e = vqeclass.vqe_e0
        title = "Ground States of Ising Hamiltonian ({0}-spins), J = {1}"
    else:
        true_e = vqeclass.true_e1
        true_gs_e = vqeclass.true_e0
        vqe_e = vqeclass.vqe_e1
        title = "Excited States of Ising Hamiltonian ({0}-spins), J = {1}"

    lams = np.linspace(0, 2 * vqeclass.Hs.J, vqeclass.Hs.n_states)
    ax = plt.subplots(2, 1, figsize=(12, 6))[1]

    ax[0].plot(lams, true_e, "--", label="True", color="red", lw=3)
    if excited:
        ax[0].plot(lams, vqe_e, ".", label="VQE", color="dodgerblue", lw=2)
        ax[0].plot(lams, vqe_e, color="dodgerblue", lw=2, alpha=0.6)
        ax[0].plot(
            lams, true_gs_e, "--", label="Ground State", color="black", lw=2, alpha=0.6
        )
    else:
        ax[0].plot(lams, vqe_e, ".", label="VQE", color="green", lw=2)
        ax[0].plot(lams, vqe_e, color="green", lw=2, alpha=0.6)
    ax[0].grid(True)
    ax[0].set_title(title.format(vqeclass.Hs.N, vqeclass.Hs.J))
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(r"$E(\lambda)$")
    ax[0].legend()

    accuracy = np.abs((true_e - vqe_e) / true_e)
    ax[1].fill_between(lams, 0.01, max(np.max(accuracy), 0.01), color="r", alpha=0.3)
    ax[1].fill_between(lams, 0.01, min(np.min(accuracy), 0), color="green", alpha=0.3)
    ax[1].axhline(y=0.01, color="r", linestyle="--")
    ax[1].scatter(lams, accuracy)
    ax[1].grid(True)
    ax[1].set_title("Accuracy of VQE")
    ax[1].set_xlabel(r"$\lambda$")
    ax[1].set_ylabel(r"$|(E_{vqe} - E_{true})/E_{true}|$")

    plt.tight_layout()


def show_VQE_annni(vqeclass, log_heatmap=False, excited=False, plot3d=True):
    """
    Shows results of a trained VQE run:
    > VQE enegies plot
    > Loss curve if VQE was trained using recycle = False
    > Final relative errors
    > Mean Squared difference between final neighbouring states
    """
    side = vqeclass.Hs.side

    # Exit if the VQE was not trained for excited states
    if excited:
        try:
            vqeclass.vqe_params1
        except:
            return

    if not excited:
        trues = np.reshape(vqeclass.true_e0, (side, side))
        preds = np.reshape(vqeclass.vqe_e0, (side, side))
    else:
        trues = np.reshape(vqeclass.true_e1, (side, side))
        preds = np.reshape(vqeclass.vqe_e1, (side, side))
        trues_gs = np.reshape(vqeclass.true_e0, (side, side))

    x = np.linspace(1, 0, side)
    y = np.linspace(0, 2, side)

    if plot3d:
        if excited:
            fig = go.Figure(
                data=[
                    go.Surface(opacity=0.2, colorscale="Reds", z=trues, x=x, y=y),
                    go.Surface(opacity=1, colorscale="Blues", z=preds, x=x, y=y),
                    go.Surface(opacity=0.6, colorscale="plasma", z=trues_gs, x=x, y=y),
                ]
            )
        else:
            fig = go.Figure(
                data=[
                    go.Surface(opacity=0.2, colorscale="Reds", z=trues, x=x, y=y),
                    go.Surface(opacity=1, colorscale="Blues", z=preds, x=x, y=y),
                ]
            )

        fig.update_layout(height=500)
        fig.show()

    accuracy = np.rot90(np.abs(preds - trues) / np.abs(trues))

    if not log_heatmap:
        colors_good = np.squeeze(
            np.dstack(
                (
                    np.dstack((np.linspace(0.3, 0, 25), np.linspace(0.8, 1, 25))),
                    np.linspace(1, 0, 25),
                )
            )
        )
        colors_bad = np.squeeze(
            np.dstack((np.dstack((np.linspace(1, 0, 100), [0] * 100)), [0] * 100))
        )
        colors = np.vstack((colors_good, colors_bad))
        cmap_acc = LinearSegmentedColormap.from_list("accuracies", colors)

        plt.imshow(accuracy, cmap=cmap_acc)
        plt.clim(0, 0.05)
        plt.colorbar(fraction=0.04)
    else:
        colors = np.squeeze(
            np.dstack(
                (
                    np.dstack((np.linspace(0, 1, 75), np.linspace(1, 0, 75))),
                    np.linspace(0, 0, 75),
                )
            )
        )
        cmap_acc = LinearSegmentedColormap.from_list("accuracies", colors)
        plt.imshow(accuracy, cmap=cmap_acc, norm=LogNorm())
        plt.colorbar(fraction=0.04)

    plt.title("Accuracies of VQE-states N={0}".format(vqeclass.Hs.N))
    plt.ylabel(r"$B/\,J_1$")
    plt.xlabel(r"$J_1/\,J_2$")

    plt.xticks(
        ticks=np.linspace(0, side - 1, 5).astype(int),
        labels=[np.round(k * 1 / 4, 2) for k in range(0, 5)],
    )
    plt.yticks(
        ticks=np.linspace(0, side - 1, 5).astype(int),
        labels=[np.round(k * 2 / 4, 2) for k in range(4, -1, -1)],
    )

    plt.tight_layout()


def show_QCNN_classification1D(qcnnclass, train_index):
    """
    Plots performance of the classifier on the whole data
    """

    circuit = qcnnclass._vqe_qcnn_circuit

    @qml.qnode(qcnnclass.device, interface="jax")
    def qcnn_circuit_prob(params_vqe, params):
        circuit(params_vqe, params)

        return qml.probs(wires=qcnnclass.N - 1)

    test_index = np.setdiff1d(np.arange(len(qcnnclass.vqe_params)), train_index)

    predictions_train = []
    predictions_test = []

    colors_train = []
    colors_test = []

    vcircuit = jax.vmap(lambda v: qcnn_circuit_prob(v, qcnnclass.params), in_axes=(0))

    predictions = vcircuit(qcnnclass.vqe_params)[:, 1]

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
    # ax[0].set_xlabel("Transverse field")
    ax[0].set_ylabel("Prediction of label II")
    ax[0].set_title("Predictions of labels; J = 1")
    ax[0].scatter(
        2 * np.sort(train_index) / len(qcnnclass.vqe_params),
        predictions_train,
        c="royalblue",
        label="Training samples",
    )
    ax[0].scatter(
        2 * np.sort(test_index) / len(qcnnclass.vqe_params),
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
    # ax[1].set_xlabel("Transverse field")
    ax[1].set_ylabel("Prediction of label II")
    ax[1].set_title("Predictions of labels; J = 1")
    ax[1].scatter(
        2 * np.sort(train_index) / len(qcnnclass.vqe_params),
        predictions_train,
        c=colors_train,
    )
    ax[1].scatter(
        2 * np.sort(test_index) / len(qcnnclass.vqe_params),
        predictions_test,
        c=colors_test,
    )


def show_compression_ANNNI(encclass, trainingpoint=False, label=False, plot3d=False):
    """
    Shows result of compression of the Anomaly Detector
    """

    side = encclass.vqe.Hs.side

    X = jnp.array(encclass.vqe_params0)

    @qml.qnode(encclass.device, interface="jax")
    def encoder_circuit(vqe_params, params):
        encclass._vqe_enc_circuit(vqe_params, params)

        return [qml.expval(qml.PauliZ(int(k))) for k in encclass.wires_trash]

    v_encoder_circuit = jax.vmap(lambda x: encoder_circuit(x, encclass.params))

    exps = (1 - np.sum(v_encoder_circuit(X), axis=1) / 4) / 2

    exps = np.rot90(np.reshape(exps, (side, side)))

    if plot3d:
        x = np.linspace(1, 0, side)
        y = np.linspace(0, 2, side)

        fig = go.Figure(data=[go.Surface(z=exps, x=x, y=y)])
        fig.update_layout(height=500)
        fig.show()
    else:
        plt.figure(figsize=(8, 6), dpi=80)
        plt.imshow(exps)

        # plt.ylabel(r'$h$', fontsize=24)
        plt.xlabel(r"$\kappa$", fontsize=24)

        x = np.linspace(1, 0, side)
        y = np.linspace(0, 2, side)

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

        getlines(qmlgen.paraanti, [0.5, 1 - 1e-5], side, "white", res=100)
        getlines(qmlgen.paraferro, [1e-5, 0.5], side, "white", res=100)

        if type(trainingpoint) == int:
            train_x = trainingpoint // side
            train_y = side - trainingpoint % side
            if train_x == 0:
                train_x += 1.5
            if train_y == side:
                train_y -= 2

            plt.scatter(
                [train_x],
                [train_y],
                marker="+",
                s=300,
                color="orangered",
                label=r"Initial state $\left|\psi\right\rangle$",
            )
            plt.ylim(side - 1, 0)
            plt.xlim(0, side - 1)

            leg = plt.legend(
                bbox_to_anchor=(1, 1),
                loc="upper right",
                fontsize=16,
                facecolor="white",
                markerscale=0.8,
                framealpha=0.9,
                title=r"AD,     $N = {0}$".format(str(encclass.vqe.Hs.N)),
                title_fontsize=16,
            )
            leg.get_frame().set_boxstyle("Square")

        if label:
            plt.figtext(0.23, 0.79, "(" + label + ")", color="black", fontsize=20)
        plt.text(
            side * 0.5,
            side * 0.4,
            "para.",
            color="black",
            fontsize=22,
            ha="center",
            va="center",
        )
        plt.text(
            side * 0.18,
            side * 0.88,
            "ferro.",
            color="white",
            fontsize=22,
            ha="center",
            va="center",
        )
        plt.text(
            side * 0.82,
            side * 0.88,
            "anti.",
            color="black",
            fontsize=22,
            ha="center",
            va="center",
        )

        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=18)


def show_QCNN_classification2D(qcnnclass):
    """
    Plots performance of the classifier on the whole data
    """

    @qml.qnode(qcnnclass.device, interface="jax")
    def qcnn_circuit_prob(params_vqe, params):
        qcnnclass._vqe_qcnn_circuit(params_vqe, params)

        if qcnnclass.n_outputs == 1:
            return qml.probs(wires=qcnnclass.N - 1)
        else:
            return [qml.probs(wires=int(k)) for k in qcnnclass.final_active_wires]

    mask1 = jnp.array(qcnnclass.vqe.Hs.model_params)[:, 1] == 0
    mask2 = jnp.array(qcnnclass.vqe.Hs.model_params)[:, 2] == 0

    ising_1, label_1 = (
        qcnnclass.vqe_params[mask1],
        qcnnclass.labels[mask1, :].astype(int),
    )
    ising_2, label_2 = (
        qcnnclass.vqe_params[mask2],
        qcnnclass.labels[mask2, :].astype(int),
    )

    vcircuit = jax.vmap(lambda v: qcnn_circuit_prob(v, qcnnclass.params), in_axes=(0))
    predictions1 = vcircuit(ising_1)
    predictions2 = vcircuit(ising_2)

    out1_p1, out2_p1, c1 = [], [], []
    for idx, pred in enumerate(predictions1):
        out1_p1.append(pred[0][1])
        out2_p1.append(pred[1][1])

        if (np.argmax(pred[0]) == label_1[idx][0]) and (
            np.argmax(pred[1]) == label_1[idx][1]
        ):
            c1.append("green")
        else:
            c1.append("red")

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    x = np.arange(int(np.sqrt(qcnnclass.n_states)))
    ax[0].grid(True)
    ax[0].scatter(x, out1_p1, c=c1)
    ax[0].set_ylim(-0.1, 1.1)
    ax[1].grid(True)
    ax[1].scatter(x, out2_p1, c=c1)
    ax[1].set_ylim(-0.1, 1.1)

    plt.show()

    out1_p2, out2_p2, c2 = [], [], []
    for idx, pred in enumerate(predictions2):
        out1_p2.append(pred[0][1])
        out2_p2.append(pred[1][1])

        if (np.argmax(pred[0]) == label_2[idx][0]) and (
            np.argmax(pred[1]) == label_2[idx][1]
        ):
            c2.append("green")
        else:
            c2.append("red")

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    x = np.arange(int(np.sqrt(qcnnclass.n_states)))
    ax[0].grid(True)
    ax[0].scatter(x, out1_p2, c=c2)
    ax[0].set_ylim(-0.1, 1.1)
    ax[1].grid(True)
    ax[1].scatter(x, out2_p2, c=c2)
    ax[1].set_ylim(-0.1, 1.1)

    plt.show()


def show_QCNN_classificationANNNI(
    qcnnclass,
    hard_thr=True,
    lines=False,
    deltaeline=[],
    train_index=[],
    label=False,
    info=False,
    morelines=False,
):

    plt.figure(figsize=(8, 6), dpi=80)

    circuit = qcnnclass._vqe_qcnn_circuit
    side = int(np.sqrt(qcnnclass.n_states))

    if hard_thr:

        @qml.qnode(qcnnclass.device, interface="jax")
        def qcnn_circuit_prob(params_vqe, params):
            circuit(params_vqe, params)

            return [qml.probs(wires=int(k)) for k in qcnnclass.final_active_wires]

        vcircuit = jax.vmap(
            lambda v: qcnn_circuit_prob(v, qcnnclass.params), in_axes=(0)
        )

        predictions = np.array(np.argmax(vcircuit(qcnnclass.vqe_params), axis=2))
        c = []
        for pred in predictions:
            if (pred == [0, 1]).all():
                c.append(0)
            elif (pred == [1, 1]).all():
                c.append(1)
            elif (pred == [1, 0]).all():
                c.append(2)
            else:
                c.append(3)

        phases = mpl.colors.ListedColormap(
            ["lightcoral", "skyblue", "black", "palegreen"]
        )
        norm = mpl.colors.BoundaryNorm(np.arange(0, 4), phases.N)
        plt.imshow(np.rot90(np.reshape(c, (side, side))), cmap=phases, norm=norm)
    else:

        @qml.qnode(qcnnclass.device, interface="jax")
        def qcnn_circuit_prob(params_vqe, params):
            circuit(params_vqe, params)

            return qml.probs([int(k) for k in qcnnclass.final_active_wires])

        vcircuit = jax.vmap(
            lambda v: qcnn_circuit_prob(v, qcnnclass.params), in_axes=(0)
        )

        predictions = np.array(vcircuit(qcnnclass.vqe_params))
        mygreen = np.array([90, 255, 100]) / 255
        myblue = np.array([50, 50, 200]) / 255
        myyellow = np.array([300, 270, 0]) / 255
        c = []

        rgb_probs = np.ndarray(shape=(side * side, 3), dtype=float)

        for i, pred in enumerate(predictions):
            rgb_probs[i] = pred[3] * mygreen + pred[1] * myblue + pred[2] * myyellow

        rgb_probs = np.rot90(np.reshape(rgb_probs, (side, side, 3)))

        plt.imshow(rgb_probs, alpha=1)

    plt.ylabel(r"$h$", fontsize=24)
    plt.xlabel(r"$\kappa$", fontsize=24)

    x = np.linspace(1, 0, side)
    y = np.linspace(0, 2, side)

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

    if len(train_index) > 0:
        x_star, y_star = [], []
        for idx in train_index:
            x_star.append(idx // side - 1)
            y_star.append(side - idx % side)
            if x_star[-1] == -1:
                x_star[-1] = x_star[-1] + 1.5
            if y_star[-1] == side:
                y_star[-1] = y_star[-1] - 1.5

        plt.scatter(
            x_star,
            y_star,
            marker="o",
            color="red",
            s=30,
            alpha=1,
            label=r"Training Points $\mathcal{{S}}_U^{14}$",
        )
        plt.ylim(side - 1, 0)
        plt.xlim(0, side - 1)

    if len(deltaeline) > 0:
        plt.plot(
            [side / 2] * int(side / 2) + (side / 2) * deltaeline,
            color="fuchsia",
            lw=2.5,
            alpha=0.9,
            label="VQE RG",
        )

    getlines(qmlgen.paraanti, [0.5, 1], side, "white", res=100)
    getlines(qmlgen.paraferro, [0, 0.5], side, "white", res=100)
    if morelines:
        getlines(qmlgen.peshel_emery, [0, 0.5], side, "cyan", res=100)
        getlines(qmlgen.b1, [0.5, 1], side, "blue", res=100)

    if label:
        plt.figtext(0.28, 0.79, "(" + label + ")", color="black", fontsize=20)

    if info:
        plt.text(
            side * 0.5,
            side * 0.4,
            "para.",
            color="black",
            fontsize=20,
            ha="center",
            va="center",
        )
        plt.text(
            side * 0.18,
            side * 0.88,
            "ferro.",
            color="white",
            fontsize=20,
            ha="center",
            va="center",
        )
        plt.text(
            side * 0.82,
            side * 0.88,
            "anti.",
            color="black",
            fontsize=20,
            ha="center",
            va="center",
        )

        leg = plt.legend(
            bbox_to_anchor=(1, 1),
            loc="upper right",
            fontsize=16,
            facecolor="white",
            markerscale=1,
            framealpha=0.9,
            title=r"QCNN,     $N = {0}$".format(str(qcnnclass.N)),
            title_fontsize=16,
        )
        leg.get_frame().set_boxstyle("Square")

        # change the line width for the legend
        for line in leg.get_lines():
            line.set_linewidth(4.0)
