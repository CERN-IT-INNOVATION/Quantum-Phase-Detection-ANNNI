import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import plotly.graph_objects as go
import pandas as pd

def show_VQE_isingchain(vqeclass, excited = False):
    """
    Shows results of a trained VQE run:
    > VQE enegies plot
    > Loss curve if VQE was trained using recycle = False
    > Final relative errors
    > Mean Squared difference between final subsequent states
    """
    # Exit if the VQE was not trained for excited states
    if excited:
        try:
            vqeclass.vqe_params1
        except:
            return
    
    if not excited:
        true_e = vqeclass.true_e0
        vqe_e  = vqeclass.vqe_e0
        vqe_params = vqeclass.vqe_params0
        title = "Ground States of Ising Hamiltonian ({0}-spins), J = {1}"
    else:
        true_e = vqeclass.true_e1
        true_gs_e = vqeclass.true_e0
        vqe_e  = vqeclass.vqe_e1 
        vqe_params = vqeclass.vqe_params1
        title = "Excited States of Ising Hamiltonian ({0}-spins), J = {1}"
        
    @qml.qnode(vqeclass.device, interface="jax")
    def q_vqe_state(vqe_params):
        vqeclass.circuit(vqe_params)

        return qml.state()
    
    j_q_vqe_state = jax.jit(q_vqe_state)
    
    lams = np.linspace(0, 2*vqeclass.Hs.J, vqeclass.n_states)
    tot_plots = 2
    fig, ax = plt.subplots(tot_plots, 1, figsize=(12, 6*tot_plots))

    ax[0].plot(lams, true_e, "--", label="True", color="red", lw=3)
    if excited:
        ax[0].plot(lams, vqe_e, ".", label="VQE", color="dodgerblue", lw=2)
        ax[0].plot(lams, vqe_e, color="dodgerblue", lw=2, alpha=0.6)
        ax[0].plot(lams, true_gs_e, "--", label="Ground State", color="black", lw=2, alpha = 0.6)
    else:
        ax[0].plot(lams, vqe_e, ".", label="VQE", color="green", lw=2)
        ax[0].plot(lams, vqe_e, color="green", lw=2, alpha=0.6)
    ax[0].grid(True)
    ax[0].set_title(title.format(vqeclass.N, vqeclass.Hs.J))
    ax[0].set_xlabel(r"$\lambda$")
    ax[0].set_ylabel(r"$E(\lambda)$")
    ax[0].legend()
    
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
    ax[1].set_title("Accuracy of VQE".format(vqeclass.N, vqeclass.Hs.J))
    ax[1].set_xlabel(r"$\lambda$")
    ax[1].set_ylabel(r"$|(E_{vqe} - E_{true})/E_{true}|$")

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

def show_VQE_annni(vqeclass, log_heatmap = False, excited = False, plot3d = True):
    """
    Shows results of a trained VQE run:
    > VQE enegies plot
    > Loss curve if VQE was trained using recycle = False
    > Final relative errors
    > Mean Squared difference between final neighbouring states
    """
    states_dist = []
    side = int(np.sqrt(vqeclass.n_states))
    
    # Exit if the VQE was not trained for excited states
    if excited:
        try:
            vqeclass.vqe_params1
        except:
            return
    
    if not excited:
        #states = vqeclass.states
        trues = np.reshape(vqeclass.true_e0,(side, side) )
        preds = np.reshape(vqeclass.vqe_e0,(side, side) )
        title = "Ground States of Ising Hamiltonian ({0}-spins), J = {1}"
    else:
        #states = vqeclass.states1
        trues = np.reshape(vqeclass.true_e1,(side, side) )
        preds = np.reshape(vqeclass.vqe_e1,(side, side) )
        trues_gs = np.reshape(vqeclass.true_e0,(side, side) )
        title = "Excited States of Ising Hamiltonian ({0}-spins), J = {1}"

    x = np.linspace(1, 0, side)
    y = np.linspace(0, 2, side)
    
    if plot3d:
        if excited:
            fig = go.Figure(data=[go.Surface(opacity=.2, colorscale='Reds', z=trues, x=x, y=y),
                      go.Surface(opacity=1, colorscale='Blues',z=preds, x=x, y=y), go.Surface(opacity=.6, colorscale='plasma',z=trues_gs, x=x, y=y)])
        else:
            fig = go.Figure(data=[go.Surface(opacity=.2, colorscale='Reds', z=trues, x=x, y=y),
                      go.Surface(opacity=1, colorscale='Blues',z=preds, x=x, y=y)])

        fig.update_layout(height=500)
        fig.show()

    accuracy = np.rot90( np.abs(preds-trues)/np.abs(trues) )

    if not log_heatmap:
        colors_good = np.squeeze( np.dstack((np.dstack((np.linspace(.3,0,25), np.linspace(.8,1,25))), np.linspace(1,0,25) )) )
        colors_bad  = np.squeeze( np.dstack((np.dstack((np.linspace(1,0,100), [0]*100)), [0]*100 )) )
        colors = np.vstack((colors_good, colors_bad))
        cmap_acc = LinearSegmentedColormap.from_list('accuracies', colors)

        plt.imshow(accuracy, cmap = cmap_acc)
        plt.clim(0,0.05)
        plt.colorbar(fraction=0.04)
    else:
        colors = np.squeeze( np.dstack((np.dstack((np.linspace(0,1,75), np.linspace(1,0,75))), np.linspace(0,0,75) )) )
        cmap_acc = LinearSegmentedColormap.from_list('accuracies', colors)
        plt.imshow(accuracy, cmap = cmap_acc, norm=LogNorm())
        plt.colorbar(fraction=0.04)

    plt.title('Accuracies of VQE-states N={0}'.format(vqeclass.N))
    plt.ylabel(r'$B/\,J_1$')
    plt.xlabel(r'$J_1/\,J_2$')

    plt.xticks(ticks=np.linspace(0,side-1,5).astype(int), labels= [np.round(k*1/4,2) for k in range(0,5)] )
    plt.yticks(ticks=np.linspace(0,side-1,5).astype(int), labels= [np.round(k*2/4,2) for k in range(4,-1,-1)])
    
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

    circuit = qcnnclass.vqe_qcnn_circuit
    
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
    #ax[0].set_xlabel("Transverse field")
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
    #ax[1].set_xlabel("Transverse field")
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
    
def show_VQE_crossfidelties(vqeclass):
    heatmap = np.zeros((vqeclass.n_states,vqeclass.n_states))
    
    for j, state1 in enumerate(vqeclass.states):
        for k, state2 in enumerate(vqeclass.states):
            heatmap[j,k] = np.square(np.abs( np.conj(state1) @ state2 ))
            
    plt.imshow(heatmap)
    plt.clim(0,1)
    plt.colorbar()
    plt.show() 
    
    print('Mean Cross Fidelty: {0}'.format(np.mean(heatmap)) )
    
    neighbouring_fidelties = []
    for state_prev, state in zip(np.array(vqeclass.states), np.array(vqeclass.states)[1:]):
        neighbouring_fidelties.append(np.square(np.abs( np.conj(state_prev) @ state )) )
        
    plt.plot(neighbouring_fidelties)
    plt.title('Fidelties beetween a state and his next one')
    plt.xlabel('State #')
    plt.ylabel('F')
    plt.show()
    
    true_states = []
    for h in vqeclass.Hs.mat_Hs:
        # Compute eigenvalues and eigenvectors
        eigval, eigvec = jnp.linalg.eigh(h)
        # Get the eigenstate to the lowest eigenvalue
        gstate = eigvec[:,jnp.argmin(eigval)]

        true_states.append(gstate)
        
    vqe_true_fidelties = []
    for state_vqe, state_true in zip(np.array(vqeclass.states), np.array(true_states)):
        vqe_true_fidelties.append(np.square(np.abs( np.conj(state_vqe) @ state_true )) )
        
    plt.plot(vqe_true_fidelties)
    plt.show()
    
def show_true_crossfidelties(H, show_energy = False):
    heatmap = np.zeros((H.n_states,H.n_states))
    true_states = []
    
    for h in H.mat_Hs:
        # Compute eigenvalues and eigenvectors
        eigval, eigvec = jnp.linalg.eigh(h)
        # Get the eigenstate to the lowest eigenvalue
        gstate = eigvec[:,jnp.argmin(eigval)]

        true_states.append(gstate)
    
    for j, state1 in enumerate(true_states):
        for k, state2 in enumerate(true_states):
            heatmap[j,k] = np.square(np.abs( np.conj(state1) @ state2 ))
            
    plt.imshow(heatmap)
    plt.clim(0,1)
    plt.colorbar()
    plt.show()
    
    print('Mean Cross Fidelty: {0}'.format(np.mean(heatmap)) )
    
    neighbouring_fidelties = []
    for state_prev, state in zip(np.array(true_states), np.array(true_states)[1:]):
        neighbouring_fidelties.append(np.square(np.abs( np.conj(state_prev) @ state )) )
        
    plt.plot(neighbouring_fidelties)
    plt.title('Fidelties beetween a state and his next one')
    plt.xlabel('State #')
    plt.ylabel('F')
    plt.show()
        
    if show_energy:
        true_E = []
        for gstate, h in zip(true_states, H.mat_Hs):
            true_E.append( np.real( np.conj(gstate) @ h @ gstate ) )
            
        plt.plot(true_E)
        plt.show()
        
def show_compression_isingchain(encclass, inject = False):
    '''
    Shows result of compression of the Anomaly Detector
    '''
    train_index = encclass.train_index

    if not inject:
        X_train = jnp.array(encclass.vqe_params[train_index])
        test_index = np.setdiff1d(np.arange(len(encclass.vqe_params)), train_index)
        X_test = jnp.array(encclass.vqe_params[test_index])

        @qml.qnode(encclass.device, interface="jax")
        def encoder_circuit(vqe_params, params):
            encclass.vqe_enc_circuit(vqe_params, params)

            return [qml.expval(qml.PauliZ(int(k))) for k in encclass.wires_trash]
    else:
        try:
            qcnnclass.psi
        except:
            psi = []
            for h in encclass.vqe.Hs.mat_Hs:
                # Compute eigenvalues and eigenvectors
                eigval, eigvec = jnp.linalg.eigh(h)
                # Get the eigenstate to the lowest eigenvalue
                gstate = eigvec[:,jnp.argmin(eigval)]

                psi.append(gstate)
            psi = jnp.array(psi)
            encclass.psi = psi
            
        X_train = jnp.array(encclass.psi[train_index])
        test_index = np.setdiff1d(np.arange(len(encclass.psi)), train_index)
        X_test = jnp.array(encclass.psi[test_index])

        @qml.qnode(encclass.device, interface="jax")
        def encoder_circuit(psi, params):
            encclass.psi_enc_circuit(psi, params)

            return [qml.expval(qml.PauliZ(int(k))) for k in encclass.wires_trash]

    v_encoder_circuit = jax.vmap(lambda x: encoder_circuit(x, encclass.params))

    exps_train = (1 - np.sum(v_encoder_circuit(X_train), axis=1) / 4) / 2
    exps_test = (1 - np.sum(v_encoder_circuit(X_test), axis=1) / 4) / 2

    plt.figure(figsize=(10, 3))
    plt.scatter(train_index, exps_train)
    plt.scatter(
        np.setdiff1d(np.arange(len(X_train) + len(X_test)), train_index),
        exps_test,
        label="Test",
    )
    plt.axvline(x=len(encclass.vqe_params) // 2, color="red", linestyle="--")
    plt.legend()
    plt.grid(True)
    
def show_compression_ANNNI(encclass, inject = False, plot3d = False):
    '''
    Shows result of compression of the Anomaly Detector
    '''
    side = int(np.sqrt(encclass.n_states))
    
    if not inject:
        X = jnp.array(encclass.vqe_params)

        @qml.qnode(encclass.device, interface="jax")
        def encoder_circuit(vqe_params, params):
            encclass.vqe_enc_circuit(vqe_params, params)

            return [qml.expval(qml.PauliZ(int(k))) for k in encclass.wires_trash]
    else:
        try:
            qcnnclass.psi
        except:
            psi = []
            for h in encclass.vqe.Hs.mat_Hs:
                # Compute eigenvalues and eigenvectors
                eigval, eigvec = jnp.linalg.eigh(h)
                # Get the eigenstate to the lowest eigenvalue
                gstate = eigvec[:,jnp.argmin(eigval)]

                psi.append(gstate)
            psi = jnp.array(psi)
            encclass.psi = psi
            
        X = jnp.array(encclass.psi)

        @qml.qnode(encclass.device, interface="jax")
        def encoder_circuit(psi, params):
            encclass.psi_enc_circuit(psi, params)

            return [qml.expval(qml.PauliZ(int(k))) for k in encclass.wires_trash]

    v_encoder_circuit = jax.vmap(lambda x: encoder_circuit(x, encclass.params))

    exps = (1 - np.sum(v_encoder_circuit(X), axis=1) / 4) / 2
    
    exps = np.rot90( np.reshape(exps, (side,side)) )
    
    if plot3d:
        x = np.linspace(1, 0, side)
        y = np.linspace(0, 2, side)

        fig = go.Figure(data=[go.Surface(z=exps, x=x, y=y)])
        fig.update_layout(height=500)
        fig.show()
    else:
        plt.imshow(exps)
        plt.colorbar()
        
def show_QCNN_classification2D(qcnnclass, inject = False):
    """
    Plots performance of the classifier on the whole data
    """
    circuit = qcnnclass.vqe_qcnn_circuit if inject == False else qcnnclass.psi_qcnn_circuit
    
    @qml.qnode(qcnnclass.device, interface="jax")
    def qcnn_circuit_prob(params_vqe, params):
        circuit(params_vqe, params)

        if qcnnclass.n_outputs == 1:
            return qml.probs(wires=self.N - 1)
        else:
            return [qml.probs(wires=int(k)) for k in qcnnclass.final_active_wires]
        
    mask1 = jnp.array(qcnnclass.vqe.Hs.model_params)[:,1] == 0
    mask2 = jnp.array(qcnnclass.vqe.Hs.model_params)[:,2] == 0

    ising_1, label_1  = qcnnclass.vqe_params[mask1], qcnnclass.labels[mask1,:].astype(int)
    ising_2, label_2  = qcnnclass.vqe_params[mask2], qcnnclass.labels[mask2,:].astype(int)

    vcircuit = jax.vmap(lambda v: qcnn_circuit_prob(v, qcnnclass.params), in_axes=(0))
    predictions1 = vcircuit(ising_1)
    predictions2 = vcircuit(ising_2)

    out1_p1, out2_p1, c1 = [], [], []
    for idx, pred in enumerate(predictions1):
        out1_p1.append(pred[0][1])
        out2_p1.append(pred[1][1])

        if (np.argmax(pred[0]) == label_1[idx][0]) and (np.argmax(pred[1]) == label_1[idx][1]):
            c1.append('green')
        else:
            c1.append('red')

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    x = np.arange(int(np.sqrt(qcnnclass.n_states)))
    ax[0].grid(True)
    ax[0].scatter(x, out1_p1, c=c1)
    ax[0].set_ylim(-.1,1.1)
    ax[1].grid(True)
    ax[1].scatter(x, out2_p1, c=c1)
    ax[1].set_ylim(-.1,1.1)

    plt.show()

    out1_p2, out2_p2, c2 = [], [], []
    for idx, pred in enumerate(predictions2):
        out1_p2.append(pred[0][1])
        out2_p2.append(pred[1][1])

        if (np.argmax(pred[0]) == label_2[idx][0]) and (np.argmax(pred[1]) == label_2[idx][1]):
            c2.append('green')
        else:
            c2.append('red')

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    x = np.arange(int(np.sqrt(qcnnclass.n_states)))
    ax[0].grid(True)
    ax[0].scatter(x, out1_p2, c=c2)
    ax[0].set_ylim(-.1,1.1)
    ax[1].grid(True)
    ax[1].scatter(x, out2_p2, c=c2)
    ax[1].set_ylim(-.1,1.1)

    plt.show()
    
def show_QCNN_classificationANNNI(qcnnclass, hard_thr = True, lines = False, deltaeline = [], train_index = [], label = False):
    circuit = qcnnclass.vqe_qcnn_circuit
    side = int(np.sqrt(qcnnclass.n_states))
    
    if hard_thr:
        @qml.qnode(qcnnclass.device, interface="jax")
        def qcnn_circuit_prob(params_vqe, params):
            circuit(params_vqe, params)
        
            return [qml.probs(wires=int(k)) for k in qcnnclass.final_active_wires]
        
        vcircuit = jax.vmap(lambda v: qcnn_circuit_prob(v, qcnnclass.params), in_axes=(0))
    
        predictions = np.array(np.argmax(vcircuit(qcnnclass.vqe_params), axis = 2))
        c = []
        for pred in predictions:
            if (pred == [0,1]).all():
                c.append(0)
            elif (pred == [1,1]).all():
                c.append(1)
            elif (pred == [1,0]).all():
                c.append(2)
            else: c.append(3)

        phases = mpl.colors.ListedColormap(["lightcoral", "skyblue",
                                            "black", "palegreen"])
        norm = mpl.colors.BoundaryNorm(np.arange(0,4), phases.N) 
        plt.imshow( np.rot90(np.reshape(c, (side, side) )), 
                                      cmap = phases, norm = norm)
    else:
        @qml.qnode(qcnnclass.device, interface="jax")
        def qcnn_circuit_prob(params_vqe, params):
            circuit(params_vqe, params)
        
            return qml.probs([int(k) for k in qcnnclass.final_active_wires])
        
        vcircuit = jax.vmap(lambda v: qcnn_circuit_prob(v, qcnnclass.params), in_axes=(0))
    
        predictions = np.array(vcircuit(qcnnclass.vqe_params) )
        
        c = []
        
        # define color map 
        color_map = {1: np.array([255, 0, 0]), # red
                     2: np.array([0, 255, 0]), # green
                     3: np.array([0, 0, 255])} # blue
        
        rgb_probs = np.ndarray(shape=(side*side, 3), dtype=float)
        
        for i, pred in enumerate(predictions):
            rgb_probs[i] = [pred[1]*255,pred[2]*255,pred[3]*255]
        rgb_probs = np.rot90(np.reshape(rgb_probs, (side,side,3)) )/255
        
        plt.imshow( rgb_probs )
        
    plt.ylabel(r'$h\,/\,J_1$')
    plt.xlabel(r'$K$')
    
    def getlines(func, xrange, side, color, res = 100):
        xs = np.linspace(xrange[0], xrange[1], res)
        ys = func(xs)
        
        plt.plot(side*xs -.5, side - ys*side/2 -.5, color = color)
        
    def B2SA(x):
        return 1.05 * np.sqrt((x-.5)*(x-.1))
    
    def ferropara(x):
        return 1 - 2*x
                    
    getlines(B2SA, [.5,1], side, 'white', res = 100)
    getlines(ferropara, [0,.5], side, 'white', res = 100)
    
    x = np.linspace(1, 0, side)
    y = np.linspace(0, 2, side)

    plt.xticks(ticks=np.linspace(0,side-1,5).astype(int), labels= [np.round(k*1/4,2) for k in range(0,5)] )
    plt.yticks(ticks=np.linspace(0,side-1,5).astype(int), labels= [np.round(k*2/4,2) for k in range(4,-1,-1)])
    
    if len(deltaeline)>0:
        plt.plot([side/2]*int(side/2) + (side/2) * deltaeline, '--', color ='black', lw= 2.5)
        
    if len(train_index)>0:
        x_star, y_star = [], []
        for idx in train_index:
            x_star.append(idx//side)
            y_star.append(side - idx%side - .5)
        plt.scatter(x_star, y_star, marker = 'o', color = 'yellow', s = 205, alpha =1)
        plt.ylim(side-1,0)
        plt.xlim(0,side-1)
    
    if label:
        plt.figtext(.33, .8, '('+label+')', color = 'white', fontsize=14)
        