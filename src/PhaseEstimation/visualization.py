import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import plotly.graph_objects as go
from tqdm.auto import tqdm

from PhaseEstimation import general as qmlgen
from PhaseEstimation import losses

from typing import List, Callable

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"]})
rc("text", usetex=True)

#  __           _______  _______ .__   __.  _______ .______          ___       __      
# /_ |         /  _____||   ____||  \ |  | |   ____||   _  \        /   \     |  |     
#  | |        |  |  __  |  |__   |   \|  | |  |__   |  |_)  |      /  ^  \    |  |     
#  | |        |  | |_ | |   __|  |  . `  | |   __|  |      /      /  /_\  \   |  |     
#  | |  __    |  |__| | |  |____ |  |\   | |  |____ |  |\  \----./  _____  \  |  `----.
#  |_| (__)    \______| |_______||__| \__| |_______|| _| `._____/__/     \__\ |_______|
                                                                                     
def getlines(
    func: Callable, xrange: List[float], side: int, color: str, res: int = 100
):
    """
    Plot function func from xrange[0] to xrange[1]
    This first function assumes your parameter ranges are
    > h     : [ 0, 2]
    > kappa : [ 0,-1]

    """
    xs = np.linspace(xrange[0], xrange[1], res)
    ys = func(xs)
    plt.plot(side * xs - 0.5, side - ys * side / 2 - 0.5, color=color, alpha=0.8)

def getlines_from_Hs(
    Hs, func: Callable, xrange: List[float], res: int = 100, **kwargs
):
    """
    Plot function func from xrange[0] to xrange[1]
    This function uses the VQE class to plot the function 
    according to the vqe ranges of its parameters
    """

    # Get information from vqeclass for plotting
    # (func needs to be resized)
    side_x = Hs.n_kappas
    side_y = Hs.n_hs
    max_x  = Hs.kappa_max

    yrange = [0, Hs.h_max]
    
    xs = np.linspace(xrange[0], xrange[1], res)
    ys = func(xs)

    ys[ys > yrange[1]] = yrange[1]
    
    corrected_xs = (side_x * xs / max_x - 0.5)

    plt.plot(corrected_xs, side_y - ys * side_y / yrange[1] - 0.5, **kwargs)

def plot_layout(Hs, pe_line, phase_lines, title):
    plt.figure(figsize=(8, 6), dpi=80)

    #plt.title("Accuracies of VQE-states N={0}".format(vqeclass.Hs.N))
    plt.ylabel(r"$h$", fontsize=24)
    plt.xlabel(r"$\kappa$", fontsize=24)
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(axis="y", labelsize=18)

    ticks_x = [-.5 , Hs.n_kappas/4 - .5, Hs.n_kappas/2 - .5 , 3*Hs.n_kappas/4 - .5, Hs.n_kappas - .5]
    ticks_y = [-.5 , Hs.n_hs/4 - .5, Hs.n_hs/2 - .5 , 3*Hs.n_hs/4 - .5, Hs.n_hs - .5]

    plt.xticks(
        ticks= ticks_x,
        labels=[np.round(k * Hs.kappa_max  / 4, 2) for k in range(0, 5)],
    )
    plt.yticks(
        ticks=ticks_y,
        labels=[np.round(k * Hs.h_max / 4, 2) for k in range(4, -1, -1)],
    )


    if pe_line:
        getlines_from_Hs(Hs, qmlgen.peshel_emery, [0, 0.5], res=100, color = "blue", alpha=1, ls = '--', dashes=(4,5), label = 'Peshel-Emery line')
        
    if phase_lines:
        getlines_from_Hs(Hs, qmlgen.paraanti, [0.5, Hs.kappa_max], res=100, color = "red", label = 'Phase-transition\n lines')
        getlines_from_Hs(Hs, qmlgen.paraferro, [0, 0.5], res=100, color = "red")
    
    if len(title) > 0:
        leg = plt.legend(
                bbox_to_anchor=(1, 1),
                loc="upper right",
                fontsize=16,
                facecolor="white",
                markerscale=1,
                framealpha=0.9,
                title=title,
                title_fontsize=16,
            )
    
    plt.tight_layout()

#  ___           __    __       ___      .___  ___.  __   __      .___________.  ______   .__   __.  __       ___      .__   __.      _______.
# |__ \         |  |  |  |     /   \     |   \/   | |  | |  |     |           | /  __  \  |  \ |  | |  |     /   \     |  \ |  |     /       |
#    ) |        |  |__|  |    /  ^  \    |  \  /  | |  | |  |     `---|  |----`|  |  |  | |   \|  | |  |    /  ^  \    |   \|  |    |   (----`
#   / /         |   __   |   /  /_\  \   |  |\/|  | |  | |  |         |  |     |  |  |  | |  . `  | |  |   /  /_\  \   |  . `  |     \   \    
#  / /_   __    |  |  |  |  /  _____  \  |  |  |  | |  | |  `----.    |  |     |  `--'  | |  |\   | |  |  /  _____  \  |  |\   | .----)   |   
# |____| (__)   |__|  |__| /__/     \__\ |__|  |__| |__| |_______|    |__|      \______/  |__| \__| |__| /__/     \__\ |__| \__| |_______/    
                                                                                                                                            
def HAM_mass_gap(Hs, phase_lines = False, pe_line = False):
    """
    Shows results of a trained VQE run:
    > VQE enegies plot
    > Loss curve if VQE was trained using recycle = False
    > Final relative errors
    > Mean Squared difference between final neighbouring states
    """
    
    sidex = Hs.n_kappas
    sidey = Hs.n_hs

    mass_gap = np.reshape(Hs.true_e1 - Hs.true_e0, (sidex, sidey) )
    
    plot_layout(Hs, phase_lines=phase_lines, pe_line=pe_line, title=r"Mass Gap,     $N = {0}$".format(str(Hs.N)))

    plt.imshow(np.rot90(mass_gap), aspect = Hs.n_kappas / Hs.n_hs)
    cbar = plt.colorbar(fraction=0.04)
    cbar.ax.tick_params(labelsize=16)

def HAM_phases_plot(Hs):
    sidex = Hs.n_kappas
    sidey = Hs.n_hs
    xs = np.linspace(0,Hs.kappa_max,sidex)
    ys = np.linspace(0,Hs.h_max,sidey)

    phases = []
    for x in xs:
        for y in ys:
            if x == 0:
                if y <= 1:
                    phases.append(0)
                else:
                    phases.append(1)
            elif x <= .5:
                if y <= qmlgen.paraferro(x):
                    phases.append(0)
                else:
                    phases.append(1)
            else:
                if y <= qmlgen.paraanti(x):
                    phases.append(2)
                else:
                    phases.append(1)
                    
    cmap = colors.ListedColormap(['palegreen', 'moccasin', 'lightblue'])
    bounds=[0,1,2,3]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plot_layout(Hs, pe_line=False, phase_lines=True, title = r"State of the art phases plot")
    plt.imshow(np.rot90(np.reshape(phases, (sidex,sidey))), cmap=cmap, norm = norm, aspect = Hs.n_kappas / Hs.n_hs)

#  ____          ____    ____  ______      _______ 
# |___ \         \   \  /   / /  __  \    |   ____|
#   __) |         \   \/   / |  |  |  |   |  |__   
#  |__ <           \      /  |  |  |  |   |   __|  
#  ___) |  __       \    /   |  `--'  '--.|  |____ 
# |____/  (__)       \__/     \_____\_____\_______|
#             
def VQE_show_isingchain(vqeclass):
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
    true_e = vqeclass.true_e0
    vqe_e = vqeclass.vqe_e0
    title = "Ground States of Ising Hamiltonian ({0}-spins), J = {1}"

    lams = np.linspace(0, 2 * vqeclass.Hs.J, vqeclass.Hs.n_states)
    ax = plt.subplots(2, 1, figsize=(12, 6))[1]

    ax[0].plot(lams, true_e, "--", label="True", color="red", lw=3)
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

def VQE_show_annni(vqeclass, log_heatmap = False, plot3d=True, phase_lines = False, pe_line = False, info = False):
    """
    Shows results of a trained VQE run:
    > VQE enegies plot
    > Loss curve if VQE was trained using recycle = False
    > Final relative errors
    > Mean Squared difference between final neighbouring states
    """
    
    sidex = vqeclass.Hs.n_kappas
    sidey = vqeclass.Hs.n_hs
    max_x = vqeclass.Hs.kappa_max
    max_y = vqeclass.Hs.h_max

    trues = np.reshape(vqeclass.Hs.true_e0, (sidex, sidey))
    preds = np.reshape(vqeclass.vqe_e0,  (sidex, sidey))

    x = np.linspace(-max_x, 0, sidex)
    y = np.linspace(0, max_y, sidey)

    if plot3d:
        fig = go.Figure(
            data=[
                # x and y needed to be swapped for it to properly show the graph
                go.Surface(opacity=0.2, colorscale="Reds", z=trues, x=y, y=x),
                go.Surface(opacity=1, colorscale="Blues", z=preds, x=y, y=x),
            ]
        )

        fig.update_layout(height=500)
        fig.show()

    plot_layout(vqeclass.Hs, pe_line=pe_line, phase_lines=phase_lines, title = r"VQE,     $N = {0}$".format(str(vqeclass.Hs.N)))

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

        plt.imshow(accuracy, cmap=cmap_acc, aspect = vqeclass.Hs.n_kappas / vqeclass.Hs.n_hs)
        plt.clim(0, 0.05)
        cbar = plt.colorbar(fraction=0.04)
        cbar.ax.tick_params(labelsize=16) 
    else:
        plt.imshow(accuracy, norm=LogNorm(), aspect = vqeclass.Hs.n_kappas / vqeclass.Hs.n_hs)
        cbar = plt.colorbar(fraction=0.04)
        cbar.ax.tick_params(labelsize=16)

def VQE_psi_truepsi_fidelity(vqeclass, phase_lines = False, pe_line = False):

    sidex = vqeclass.Hs.n_kappas
    sidey = vqeclass.Hs.n_hs

    @qml.qnode(vqeclass.device, interface="jax")
    def q_vqe_state(vqe_params):
        vqeclass.circuit(vqe_params)

        return qml.state()

    jv_fidelity = jax.jit(lambda true, pars: losses.vqe_fidelities(true, pars, q_vqe_state))
    
    fidelity_map = jv_fidelity(vqeclass.Hs.true_psi0, vqeclass.vqe_params0) 
    fidelity_map = np.reshape(fidelity_map, (sidex, sidey))

    plt.figure(figsize=(8, 6), dpi=80)
    plot_layout(vqeclass.Hs, phase_lines=phase_lines, pe_line=pe_line, title=r"Fidelities,     $N = {0}$".format(str(vqeclass.Hs.N)))
    plt.imshow(np.rot90(fidelity_map), aspect = vqeclass.Hs.n_kappas / vqeclass.Hs.n_hs)
    cbar = plt.colorbar(fraction=0.04)
    cbar.ax.tick_params(labelsize=16) 

def VQE_fidelity_slice(vqeclass, slice_value, axis = 0, truestates = False):
    vqeclass.Hs.show_phasesplot()
    
    sidey, ymax = vqeclass.Hs.n_hs, vqeclass.Hs.h_max
    sidex, xmax = vqeclass.Hs.n_kappas, vqeclass.Hs.kappa_max

    ticks_x = [-.5 , vqeclass.Hs.n_kappas/4 - .5, vqeclass.Hs.n_kappas/2 - .5 , 3*vqeclass.Hs.n_kappas/4 - .5, vqeclass.Hs.n_kappas - .5]
    ticks_y = [-.5 , vqeclass.Hs.n_hs/4 - .5, vqeclass.Hs.n_hs/2 - .5 , 3*vqeclass.Hs.n_hs/4 - .5, vqeclass.Hs.n_hs - .5]

    if axis == 0:
        plt.axhline(y = sidey - slice_value*sidey/ymax - .5, color='blue', lw=2)
    else:
        plt.axvline(x = slice_value*sidex/xmax - .5, color='blue', lw=2)

    plt.show()

    def create_confusion_matrix(vectors):
        dimvec = len(vectors)
        c_matrix = np.zeros((dimvec, dimvec))

        for i in range(dimvec):
            for j in range(dimvec):
                c_matrix[i,j] = np.square( np.real(vectors[i] @ np.conj(vectors[j]) ) )

        plt.imshow(c_matrix, origin = 'lower')
        
        return c_matrix

    plt.figure(figsize=(8, 6), dpi=80)
    
    # H : 2 = index : side -> index = H * side / 2
    starting_index = slice_value * vqeclass.Hs.n_hs / vqeclass.Hs.h_max
    
    #print(indexes)
    if axis == 0:
        indexes = np.arange(starting_index,sidex*sidey,sidey).astype(int)
        print(indexes)
        print(len(indexes))
        plt.xticks(
        ticks= ticks_x,
        labels=[np.round(k * vqeclass.Hs.kappa_max  / 4, 2) for k in range(0, 5)],
        )
        plt.yticks(
            ticks=ticks_x,
            labels=[np.round(k * vqeclass.Hs.kappa_max  / 4, 2) for k in range(0, 5)],
        )
        plt.ylabel(r"$\kappa$", fontsize=24)
        plt.xlabel(r"$\kappa$", fontsize=24)
        title = f'h = {slice_value}'
    else:
        indexes = np.arange(int(sidex*slice_value)*sidey,sidex*(int(sidey*slice_value))+sidex,1).astype(int)
        print(indexes)
        print(len(indexes))
        plt.xticks(
        ticks= ticks_y,
        labels=[np.round(k * vqeclass.Hs.h_max / 4, 2) for k in range(4, -1, -1)],
        )
        plt.yticks(
            ticks=ticks_y,
            labels=[np.round(k * vqeclass.Hs.h_max / 4, 2) for k in range(4, -1, -1)],
        )
        plt.ylabel(r"$h$", fontsize=24)
        plt.xlabel(r"$h$", fontsize=24)
        title = f'k = {slice_value}'
    
    if truestates:
        try:
            vqeclass.Hs.true_psi0
        except:
            vqeclass.Hs.add_true()
        confusion = create_confusion_matrix(np.array(vqeclass.Hs.true_psi0)[indexes])
    else:
        @qml.qnode(vqeclass.device, interface="jax")
        def q_vqe_state(vqe_params):
            vqeclass.circuit(vqe_params)

            return qml.state()

        
        vqe_psi0 = jax.jit(jax.vmap(q_vqe_state, in_axes=(0)))(vqeclass.vqe_params0[indexes])
        confusion = create_confusion_matrix(vqe_psi0)

    leg = plt.legend(
            bbox_to_anchor=(1, 1),
            loc="upper right",
            fontsize=16,
            facecolor="white",
            markerscale=1,
            framealpha=0.9,
            title=title,
            title_fontsize=16,
        )

    cbar = plt.colorbar(fraction=0.04)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()

#   _  _      ____   _____ _   _ _   _ 
#  | || |    / __ \ / ____| \ | | \ | |
#  | || |_  | |  | | |    |  \| |  \| |
#  |__   _| | |  | | |    | . ` | . ` |
#     | |_  | |__| | |____| |\  | |\  |
#     |_(_)  \___\_\\_____|_| \_|_| \_|
                                                   
def QCNN_classification_ising(qcnnclass, train_index):
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

def QCNN_classification_ANNNI_marginal(qcnnclass):
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
    
    ising_1, label_1, x1 = (
        qcnnclass.vqe_params[mask1],
        qcnnclass.labels[mask1, :].astype(int),
        np.arange( len(mask1[mask1 == True]) )
    )
    ising_2, label_2, x2 = (
        qcnnclass.vqe_params[mask2],
        qcnnclass.labels[mask2, :].astype(int),
        np.arange( len(mask2[mask2 == True]) )
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

    ax[0].grid(True)
    ax[0].scatter(x1, out1_p1, c=c1)
    ax[0].set_ylim(-0.1, 1.1)
    ax[1].grid(True)
    ax[1].scatter(x1, out2_p1, c=c1)
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

    ax[0].grid(True)
    ax[0].scatter(x2, out1_p2, c=c2)
    ax[0].set_ylim(-0.1, 1.1)
    ax[1].grid(True)
    ax[1].scatter(x2, out2_p2, c=c2)
    ax[1].set_ylim(-0.1, 1.1)

    plt.show()

def QCNN_classification_ANNNI(
    qcnnclass,
    hard_thr=True,
    label=False,
    info=False,
):

    plt.figure(figsize=(8, 6), dpi=80)
    plot_layout(qcnnclass.vqe.Hs, False, True, r"QCNN,     $N = {0}$".format(str(qcnnclass.N)))
    
    circuit = qcnnclass._vqe_qcnn_circuit
    sidex = qcnnclass.vqe.Hs.n_kappas
    sidey = qcnnclass.vqe.Hs.n_hs

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
        plt.imshow(np.rot90(np.reshape(c, (sidex, sidey))), cmap=phases, norm=norm, aspect = qcnnclass.vqe.Hs.n_kappas / qcnnclass.vqe.Hs.n_hs)
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

        rgb_probs = np.ndarray(shape=(sidex * sidey, 3), dtype=float)

        for i, pred in enumerate(predictions):
            rgb_probs[i] = pred[3] * mygreen + pred[1] * myblue + pred[2] * myyellow

        rgb_probs = np.rot90(np.reshape(rgb_probs, (sidex, sidey, 3)))

        plt.imshow(rgb_probs, alpha=1, aspect = qcnnclass.vqe.Hs.n_kappas / qcnnclass.vqe.Hs.n_hs)

    if label:
        plt.figtext(0.28, 0.79, "(" + label + ")", color="black", fontsize=20)

    if info:
        # Only for (x,y)=(1,2) parameter space Hamiltonians
        # TODO: add labels for whatever hamiltonian
        if (qcnnclass.vqe.Hs.h_max == 2) and (qcnnclass.vqe.Hs.kappa_max == 1):
            plt.text(
                sidex * 0.5,
                sidey * 0.4,
                "para.",
                color="black",
                fontsize=20,
                ha="center",
                va="center",
            )
            plt.text(
                sidex * 0.18,
                sidey * 0.88,
                "ferro.",
                color="white",
                fontsize=20,
                ha="center",
                va="center",
            )
            plt.text(
                sidex * 0.82,
                sidey * 0.88,
                "anti.",
                color="black",
                fontsize=20,
                ha="center",
                va="center",
            )
#  _____    ______                     _           
# | ____|  |  ____|                   | |          
# | |__    | |__   _ __   ___ ___   __| | ___ _ __ 
# |___ \   |  __| | '_ \ / __/ _ \ / _` |/ _ \ '__|
#  ___) |  | |____| | | | (_| (_) | (_| |  __/ |   
# |____(_) |______|_| |_|\___\___/ \__,_|\___|_|   
                               
def ENC_show_compression_ANNNI(encclass, trainingpoint=False, label=False, plot3d=False):
    """
    Shows result of compression of the Anomaly Detector
    """

    sidex = encclass.vqe.Hs.n_kappas
    sidey = encclass.vqe.Hs.n_hs
    max_x = encclass.vqe.Hs.kappa_max
    max_y = encclass.vqe.Hs.h_max
    x = np.linspace(-max_x, 0, sidex)
    y = np.linspace(0, max_y, sidey)

    X = jnp.array(encclass.vqe_params0)

    @qml.qnode(encclass.device, interface="jax")
    def encoder_circuit(vqe_params, params):
        encclass._vqe_enc_circuit(vqe_params, params)

        return [qml.expval(qml.PauliZ(int(k))) for k in encclass.wires_trash]

    v_encoder_circuit = jax.vmap(lambda p: encoder_circuit(p, encclass.params))

    exps = (1 - np.sum(v_encoder_circuit(X), axis=1) / 4) / 2

    exps = np.rot90(np.reshape(exps, (sidex, sidey)))

    if plot3d:
        fig = go.Figure(data=[go.Surface(z=exps, x=x, y=y)])
        fig.update_layout(height=500)
        fig.show()

    plt.figure(figsize=(8, 6), dpi=80)
    plot_layout(encclass.vqe.Hs, pe_line=True, phase_lines=True, title='')
    plt.imshow(exps, aspect = encclass.vqe.Hs.n_kappas / encclass.vqe.Hs.n_hs)
    if type(trainingpoint) == int:
        train_x = trainingpoint // sidex
        train_y = sidey - trainingpoint % sidey
        if train_x == 0:
            train_x += 1
            print(train_x)
        if train_y == sidey:
            train_y -= 2

        plt.scatter(
            [train_x],
            [train_y],
            marker="+",
            s=300,
            color="orangered",
            label=r"Initial state $\left|\psi\right\rangle$",
        )

    if label:
        plt.figtext(0.23, 0.79, "(" + label + ")", color="black", fontsize=20)
    plt.text(
        sidex * 0.5,
        sidey * 0.4,
        "para.",
        color="black",
        fontsize=22,
        ha="center",
        va="center",
    )
    plt.text(
        sidex * 0.18,
        sidey * 0.88,
        "ferro.",
        color="white",
        fontsize=22,
        ha="center",
        va="center",
    )
    plt.text(
        sidex * 0.82,
        sidey * 0.88,
        "anti.",
        color="black",
        fontsize=22,
        ha="center",
        va="center",
    )
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
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
