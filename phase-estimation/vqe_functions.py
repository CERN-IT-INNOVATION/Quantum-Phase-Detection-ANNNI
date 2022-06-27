### IMPORTS ###
# Quantum libraries:
import pennylane as qml
from pennylane import numpy as np

# Plotting
from matplotlib import pyplot as plt

# Other
import copy
import tqdm # Pretty progress bars
import joblib # Writing and loading
from noisyopt import minimizeSPSA

import multiprocessing
##############

#   _    
#  / |   
#  | |   
#  | |_  
#  |_(_) 

def qml_build_H(N, lam, J, ring = False):
    '''
    Set up Hamiltonian: 
            H = lam*Σsigma^i_z - J*Σsigma^i_x*sigma^{i+1}_ 
    '''
    # Interaction of spins with magnetic field
    H = + lam * qml.PauliZ(0)
    for i in range(1,N):
        H = H + lam * qml.PauliZ(i)
        
    # Interaction between spins:
    for i in range(0,N-1):
        H = H + J*(-1)*( qml.PauliX(i) @ qml.PauliX(i+1) )
    
    # Ring
    if ring:
        # Create interaction between first and last qubit
        H = H + J*(-1)*( qml.PauliX(N-1) @ qml.PauliX(0) )
    
    return H


#  ____     
# |___ \    
#   __) |   
#  / __/ _  
# |_____(_) Circuit functions

def circuit_block_1(N, param, index = 0, p_noise = 0):
    '''
    RX + RY
    
    shift_invariance = 0: Each wire permorms independent rotations.
    shift_invariance = 1: Same rotation for odd and even wires,
                         expect for the first and last that have unique
                         rotation.
    shift_invariance = 2: Same rotation for each wire
    '''
    noise = True
    if p_noise == 0: noise = False # Remove BitFlip and PhaseFlip if we are not using default.mixed
    
    # Apply RX and RY to each wire:
    for spin in range(N):
        qml.RY(param[index + spin],     wires = spin)
        if noise: qml.PhaseFlip(p_noise, wires = spin); qml.BitFlip(p_noise, wires = spin)
        qml.RX(param[index + N + spin], wires = spin)
        if noise: qml.PhaseFlip(p_noise, wires = spin); qml.BitFlip(p_noise, wires = spin)
            
    return index + 2*N
    
def circuit_block_2(N, param, index = 0, p_noise = 0):
    '''
    RY
    
    shift_invariance = 0: Each wire permorms independent rotations.
    shift_invariance = 1: Same rotation for odd and even wires,
                         expect for the first and last that have unique
                         rotation.
    shift_invariance = 2: Same rotation for each wire
    '''
    noise = True
    if p_noise == 0: noise = False # Remove BitFlip and PhaseFlip if we are not using default.mixed
    
    # Apply RX and RY to each wire:
    for spin in range(N):
        #qml.RZ(param[index + N + spin],   wires = spin)
        qml.RY(param[index + spin],   wires = spin)
        if noise: qml.PhaseFlip(p_noise, wires = spin); qml.BitFlip(p_noise, wires = spin)
            
    return index + N

def circuit_entanglement(N, params, index, p_noise_ent = 0, going_down = True):
    # Apply entanglement to the neighbouring spins
    noise = True
    if p_noise_ent == 0: noise = False # Remove BitFlip and PhaseFlip if we are not using default.mixed
    
    if going_down:
        for spin in range(0,N-1):
            qml.CNOT(wires = [spin, spin+1])
            #qml.RX(params[index+spin], wires = spin+1)
            
            if noise: qml.PhaseFlip(p_noise_ent, wires = spin+1); qml.BitFlip(p_noise_ent, wires = spin+1)
            
    else:
        for spin in range(N-1,0,-1):
            qml.CNOT(wires = [spin, spin-1])
            #qml.RX(params[index+spin], wires = spin-1)
            
            if noise: qml.PhaseFlip(p_noise_ent, wires = spin-1); qml.BitFlip(p_noise_ent, wires = spin-1)
            
    return index

def vqe_circuit(N, params, p_noise = 0, p_noise_ent = 0):
    index = circuit_block_1(N, params, p_noise = p_noise)
    qml.Barrier()
    index = circuit_entanglement(N, params, index, p_noise_ent)
    qml.Barrier()
    index = circuit_block_1(N, params, index, p_noise = p_noise)
    qml.Barrier()
    index = circuit_entanglement(N, params, index, p_noise_ent)
    qml.Barrier()
    index = circuit_block_2(N, params,  index, p_noise = p_noise)
    
    return index


#  _____   
# |___ /   
#   |_ \   
#  ___) |  
# |____(_) Learning functions
    
def train(step_size, n_epochs, N, J, l_steps, device, vqe_circuit_fun, optimizer = 'adam', random_shift = 0, p_noise = 0, p_noise_ent = 0, 
          circuit = False, plots = False, prepare_states = False, preplots = False, prep_step_size = False, first_prep_epoch = 50, prep_epochs = 10,
          parameter_info = True, cutoff_value = 0.01, pretrained = []):
    
    lams = np.linspace(0, 2*J, l_steps)
    # define the wrapper update function to be called inside a multiprocessing map 
    # as a global variable to avoid the pickle error with the Jupyter Kernel
    global wrapped_update
    
    '''
    step_size    = learning rate
    random_shift = random shift of parameters for SPSA gradient evaluation
    n_epochs     = # epochs for the other lambdas
    lams         = Array of intensities of magnetic field
    N            = Number of spins of the system
    '''
    
    if parameter_info:
        print('+--- PARAMETERS ---+')
        print('step_size      = {0} (Step size of the optimizer)'.format(step_size))
        if prepare_states:
            print('prep_step_size = {0} (Step size in the preparing state phase)'.format(prep_step_size) )
        if optimizer.lower() == 'spsa':
            print('random_shift   = {0} (Random shift of parameters of the optimizer)'.format(random_shift))
        print('n_epochs       = {0} (# epochs for the other GSs)'.format(n_epochs))
        if prepare_states:
            print('prep_epochs    = {0} (# epochs in the preparing state phase)'.format(prep_epochs) )
        print('N              = {0} (Number of spins of the system)'.format(N))
        print('cutoff_value   = {0} (Cuttoff Value in relative error to stop training)\n'.format(cutoff_value) )

    
    @qml.qnode(device)
    def vqe_cost_fn(vqe_params, N, H, p_noise = 0, p_noise_ent = 0):
        vqe_circuit_fun(N, vqe_params, p_noise, p_noise_ent)

        # return <psi|H|psi>
        return qml.expval(H)

    @qml.qnode(device)
    def density_matrices(vqe_params, N, p_noise = 0, p_noise_ent = 0):
        vqe_circuit_fun(N, vqe_params, p_noise, p_noise_ent)
        
        # return |psi><psi|
        return qml.state()
    
    # circuit functions returns the number of parameters needed for the circuit itself
    n_params = vqe_circuit_fun(N, [0]*1000)
    
    # Prepare initial parameters randomly for each datapoint/state
    params = []
    for _ in lams:
        param = np.random.rand(n_params)
        params.append(param)
        
    if circuit:
        # Display the circuit
        print('+--- CIRCUIT ---+')
        drawer = qml.draw(vqe_cost_fn)
        print(drawer(np.arange(n_params), N, H = qml_build_H(N, 0 ,0)))
    
    # For each lamda create optimizer and H
    Hs   = []
    opts = []
    energy_err  = [0]*(len(lams))
    true_e = []
    
    for i, lam in enumerate(lams):
        Hs.append(qml_build_H(N, float(lam), float(J) ))
        true_e.append(np.min(qml.eigvals(Hs[i])) )
    
    if optimizer.lower() == 'adam':
        MSE = []
        for lam in lams:
            opts.append(qml.AdamOptimizer(stepsize=step_size))
            
        def mp_update_params(idx, params, N, vqe_cost_fn, opts, Hs, true_e, p_noise, p_noise_ent, epoch):
            '''
            Update function to be called inside the training function
            idx              = index to update: idx in range(len(lambdas))
            N                = number of wires/spin
            shift_invariance = type of circuit
            opts             = array of optimizers
            Hs               = array of Hamiltonians
            p_noise          = noise after R-gates
            p_noise_ent      = noise after entanglements
            '''
            param = params[idx]
            H = Hs[idx]
            opt = opts[idx]

            cost_fn = lambda v: vqe_cost_fn(v, N, H, p_noise, p_noise_ent)
            
            param_next, energy = opt.step_and_cost(cost_fn, param)
            
            return param_next, opt, (energy - true_e[idx] )**2
        
        if prepare_states:
            print('\nPreparing states:')

            # We prepare the states of every 3 parameters, the others are copied from the 
            # previous ones
            prep_progress = tqdm.tqdm(np.arange(0, len(lams), 1))

            for prep_l in prep_progress:
                if prep_l == 0:
                    prep_params = params[0]
                else:
                    prep_params = params[prep_l - 1]

                opt = qml.AdamOptimizer(stepsize=prep_step_size)
                H = qml_build_H(N, float(lams[prep_l]), J)
                cost_fn = lambda v: vqe_cost_fn(v, N, H, p_noise, p_noise_ent)

                # The first one is given more epochs being the one starting
                # from totally random parameters
                prep_epoch = first_prep_epoch if prep_l == 0 else prep_epochs

                # Actual VQE algorithm for a datapoint/state
                for epoch in range(prep_epoch):
                    prep_params, _ = opt.step_and_cost(cost_fn, prep_params)

                params[prep_l] = prep_params
                # The following two are copied of the state just found
                #if prep_l+1 < len(lams): params[prep_l+1] = prep_params
                #if prep_l+2 < len(lams): params[prep_l+2] = prep_params

            # The VQE performance can be plotted to see how close to the result
            # it is before the actual training
            if preplots:
                fig, ax = plt.subplots(1, 1, figsize=(9,4))
                vqe_e  = []
                for i, lam in enumerate(lams):
                    cost_fn = lambda v: vqe_cost_fn(v, N, Hs[i], p_noise, p_noise_ent)
                    vqe_e.append(cost_fn(params[i]) )

                ax.plot(lams, true_e, '--', label='True', color='red', lw = 2)
                ax.plot(lams, vqe_e, '.', label='VQE', color='green', lw = 2)
                ax.plot(lams, vqe_e, color='green', lw = 2, alpha=0.6)
                ax.grid(True)
                ax.set_title('Prepared - Ground States of Ising Hamiltonian ({0}-spins), J = {1}'.format(N,J))
                ax.set_xlabel(r'$\lambda$')
                ax.set_ylabel(r'$E(\lambda)$')
                ax.legend()

                plt.show()
        
        # If the VQE parameters are already pretrained we just copy 
        # the input 'pretrained' as the learning parameters
        elif len(pretrained) > 0:
            params = copy.copy(pretrained)

        progress = tqdm.tqdm(range(n_epochs))

        # IDEA in active_points:
        #   train only the datapoints/states that 
        #   have a relative error greater than CUTOFF
        active_points = np.arange(len(lams))
        
        for epoch in progress:
            # In the happy scenario all points have a relative error below CUTOFF
            # we can just stop the training aswell
            if len(active_points) == 0:
                n_epochs = epoch
                break

            def wrapped_update(idx):
                return mp_update_params(idx, params, N, vqe_cost_fn, opts, Hs, true_e, p_noise, p_noise_ent, epoch)

            p = multiprocessing.Pool()
            with p: rdata = p.map(wrapped_update, active_points)

            cutoffs = [] # Array of the new datapoints that have a relative error below CUTOFF
            for k, l in enumerate(active_points):
                params[l], opts[l], energy_err[l] = rdata[k]

                if np.sqrt(energy_err[l])/np.abs(true_e[l]) < cutoff_value:
                    cutoffs.append(l)

            # Remove all the datapoints from active_points contained in cutoff
            for cutoff in cutoffs:
                active_points = np.delete(active_points, np.where(active_points == cutoff))

            # Not actually the MSE since it is computed on just the active_points
            MSE.append(np.mean(energy_err))

            progress.set_description('Cost: {0} | Active states: {1}'.format(MSE[-1], len(active_points)) )

        vqe_e  = []
        for i, lam in enumerate(lams):
            cost_fn = lambda v: vqe_cost_fn(v, N, Hs[i], p_noise, p_noise_ent)
            vqe_e.append(cost_fn(params[i]) )

    elif optimizer.lower() == 'spsa':
        MSEs = [[0]]*(len(lams))
        def mp_train_param_SPSA(idx, params, N, vqe_cost_fn, Hs, true_e, p_noise, p_noise_ent, epochs, step_size, random_shift, MSEs):
            param = params[idx]
            H = Hs[idx]
            MSE = MSEs[idx]
            
            cost_fn = lambda v: vqe_cost_fn(v, N, H, p_noise, p_noise_ent)

            def callback_SPSA(v):
                cost_val = cost_fn(v)
                MSE.append( (true_e[idx] - cost_val)**2 )

            res = minimizeSPSA(cost_fn,
                               x0=param,
                               niter=epochs,
                               paired=False,
                               c=random_shift,
                               a=step_size,
                               callback = callback_SPSA)

            return res.x, res.fun, MSE
    
        # STATES PREPARATION:
        # Preparing initial parameters recycling thetas

        # If pre_step_size is left as false we set it
        # as step_size
        if not prep_step_size:
            prep_step_size = step_size

        if prepare_states:
            print('\nPreparing states:')

            # We prepare the states of every 3 parameters, the others are copied from the 
            # previous ones
            prep_progress = tqdm.tqdm(np.arange(0, len(lams), 3))
            vqe_e  = []

            for prep_l in prep_progress:
                if prep_l == 0:
                    prep_params = params[0]
                else:
                    prep_params = params[prep_l - 1]

                H = qml_build_H(N, float(lams[prep_l]), J)
                cost_fn = lambda v: vqe_cost_fn(v, N, H, p_noise, p_noise_ent)

                # The first one is given more epochs being the one starting
                # from totally random parameters
                prep_epochs = 5*prep_epochs if prep_l == 0 else prep_epochs

                res = minimizeSPSA(cost_fn,
                                   x0=prep_params,
                                   niter=prep_epochs,
                                   paired=False,
                                   c=random_shift,
                                   a=prep_step_size)

                params[prep_l] = res.x
                MSEs[prep_l]   = [(res.fun - true_e[prep_l])**2]

                # The following two are copied of the state just found
                if prep_l+1 < len(lams):
                    params[prep_l+1] = res.x
                    cost_fn = lambda v: vqe_cost_fn(v, N, H, p_noise, p_noise_ent)
                    MSEs[prep_l + 1] = [(cost_fn(res.x) - true_e[prep_l+1])**2]
                if prep_l+2 < len(lams):
                    params[prep_l+2] = res.x
                    cost_fn = lambda v: vqe_cost_fn(v, N, H, p_noise, p_noise_ent)
                    MSEs[prep_l + 2] = [(cost_fn(res.x) - true_e[prep_l+2])**2]

            # The VQE performance can be plotted to see how close to the result
            # it is before the actual training
            if preplots:
                fig, ax = plt.subplots(1, 1, figsize=(9,4))
                for i, lam in enumerate(lams):
                    cost_fn = lambda v: vqe_cost_fn(v, N, Hs[i], p_noise, p_noise_ent)
                    vqe_e.append(cost_fn(params[i]) )

                ax.plot(lams, true_e, '--', label='True', color='red', lw = 2)
                ax.plot(lams, vqe_e, '.', label='VQE', color='green', lw = 2)
                ax.plot(lams, vqe_e, color='green', lw = 2, alpha=0.6)
                ax.grid(True)
                ax.set_title('Prepared - Ground States of Ising Hamiltonian ({0}-spins), J = {1}'.format(N,J))
                ax.set_xlabel(r'$\lambda$')
                ax.set_ylabel(r'$E(\lambda)$')
                ax.legend()

                plt.show()

        elif len(pretrained) > 0:
            params = copy.copy(pretrained)

        def wrapped_update(idx):
            return mp_train_param_SPSA(idx, params, N, vqe_cost_fn, Hs, true_e, 
                                       p_noise, p_noise_ent, n_epochs, step_size, random_shift, MSEs)

        p = multiprocessing.Pool()
        with p: rdata = p.map(wrapped_update, np.arange(len(lams)) )

        vqe_e = []
        MSEs = []
        for l, lam in enumerate(lams):
            params[l] = rdata[l][0]
            cost = rdata[l][1]
            MSEs.append(rdata[l][2])
            vqe_e.append(cost)

        MSE = np.mean(MSEs,axis=0)[1:]

    if plots:
        fig, ax = plt.subplots(4, 1, figsize=(12,18.6))
                          
        ax[0].plot(lams, true_e, '--', label='True', color='red', lw = 2)
        ax[0].plot(lams, vqe_e, '.', label='VQE', color='green', lw = 2)
        ax[0].plot(lams, vqe_e, color='green', lw = 2, alpha=0.6)
        ax[0].grid(True)
        ax[0].set_title('Ground States of Ising Hamiltonian ({0}-spins), J = {1}'.format(N,J))
        ax[0].set_xlabel(r'$\lambda$')
        ax[0].set_ylabel(r'$E(\lambda)$')
        ax[0].legend()
        
        ax[1].plot(range(n_epochs), MSE, '.', color='orange', ms = 7 )
        ax[1].plot(range(n_epochs), MSE, color='orange', alpha=0.4)
        ax[1].set_title('Convergence of VQE')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('MSE')
        ax[1].grid(True)
        ax[1].axhline(y=0, color='r', linestyle='--')
        
        true_e = np.array(true_e)
        vqe_e = np.array(vqe_e)
        accuracy = np.abs((true_e-vqe_e)/true_e)
        ax[2].fill_between(lams, 0.01, max(np.max(accuracy),0.01), color = 'r', alpha = 0.3 )
        ax[2].fill_between(lams, .01, min(np.min(accuracy),0), color = 'green', alpha = 0.3 )
        ax[2].axhline(y=0.01, color='r', linestyle='--')
        ax[2].scatter(lams, accuracy)
        ax[2].grid(True)
        ax[2].set_title('Accuracy of VQE'.format(N,J))
        ax[2].set_xlabel(r'$\lambda$')
        ax[2].set_ylabel(r'$|(E_{vqe} - E_{true})/E_{true}|$')
        
        rho_dist = []
        for rho in range(l_steps - 1):
            rho_dist.append( np.real( np.mean( np.square(density_matrices(params[rho], N, 0) - density_matrices(params[rho+1], N, 0)) ) ) )
            
        ax[3].set_title('Mean square distance between consecutives density matrices')
        ax[3].plot(np.linspace(0,2*J, num=l_steps-1), rho_dist, '-o')
        ax[3].grid(True)
        ax[3].axvline(x=J, color='gray', linestyle='--')
        ax[3].set_xlabel(r'$\lambda$')
        
        plt.tight_layout()
        
    ys = []
    for l in lams:
        ys.append(0) if l <= J else ys.append(1)
        
    return vqe_e, MSE, params, ys   

#  _  _     
# | || |    
# | || |_   
# |__   _|  
#    |_|(_) Visualization      
def show_train_plots(data, N, J, device, vqe_circuit_fun, lsteps = 100):
    
    @qml.qnode(device)
    def vqe_cost_fn(vqe_params, N, H, p_noise = 0, p_noise_ent = 0):
        vqe_circuit_fun(N, vqe_params, p_noise, p_noise_ent)

        # return <psi|H|psi>
        return qml.expval(H)
    
    lams = np.linspace(0,2*J, lsteps)
    
    true_e = []
    vqe_e = []
    for i, l in enumerate(lams):
        circ_params = data[i][0]
        
        H = qml_build_H(N, float(l), float(J))
        true_e.append(np.min(qml.eigvals(H)))
        vqe_e.append(vqe_cost_fn(circ_params, N, H) )
        
    fig, ax = plt.subplots(2, 1, figsize=(12,9.3))
                          
    ax[0].plot(lams, true_e, '--', label='True', color='red', lw = 2)
    ax[0].plot(lams, vqe_e, '.', label='VQE', color='green', lw = 2)
    ax[0].plot(lams, vqe_e, color='green', lw = 2, alpha=0.6)
    ax[0].grid(True)
    ax[0].set_title('Ground States of Ising Hamiltonian ({0}-spins), J = {1}'.format(N,J))
    ax[0].set_xlabel(r'$\lambda$')
    ax[0].set_ylabel(r'$E(\lambda)$')
    ax[0].legend()

    true_e = np.array(true_e)
    vqe_e = np.array(vqe_e)
    accuracy = np.abs((true_e-vqe_e)/true_e)
    ax[1].fill_between(lams, 0.01, max(np.max(accuracy),0.01), color = 'r', alpha = 0.3 )
    ax[1].fill_between(lams, .01, min(np.min(accuracy),0), color = 'green', alpha = 0.3 )
    ax[1].axhline(y=0.01, color='r', linestyle='--')
    ax[1].scatter(lams, accuracy)
    ax[1].grid(True)
    ax[1].set_title('Accuracy of VQE'.format(N,J))
    ax[1].set_xlabel(r'$\lambda$')
    ax[1].set_ylabel(r'$|(E_{vqe} - E_{true})/E_{true}|$')

    plt.tight_layout()