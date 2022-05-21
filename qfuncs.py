### IMPORTS ###
# Quantum libraries:
import pennylane as qml
from pennylane import numpy as np

# Plotting
from matplotlib import pyplot as plt

# Other
import copy
from tqdm.notebook import tqdm # Pretty progress bars
from IPython.display import Markdown, display # Better prints
##############

#     88       
#   ,d88       
# 888888       
#     88       
#     88       
#     88       
#     88  888  
#     88  888  Arrays


             
             

#                   
#  ad888888b,       
# d8"     "88       
#         a8P       
#      ,d8P"        
#    a8P"           
#  a8P'             
# d8"          888  
# 88888888888  888  Ising Hamiltonian Systems

### SET UP PENNYLANE HAMILTONIAN ###
def qml_build_H(N, lam, J):
    '''
    Set up Hamiltonian: 
            H = lam*Σsigma^i_z - J*Σsigma^i_x*sigma^{i+1}_ 
    '''
    # Interaction of spins with magnetic field
    H = lam * qml.PauliZ(0)
    for i in range(1,N):
        H = H + lam * qml.PauliZ(i)
        
    # Interaction between spins:
    for i in range(0,N-1):
        H = H + J*(-1)*( qml.PauliX(i) @ qml.PauliX(i+1) )
        
    return H

                  
#  ad888888b,       
# d8"     "88       
#         a8P       
#      aad8"        
#      ""Y8,        
#         "8b       
# Y8,     a88  888  
#  "Y888888P'  888  Hamiltonians (General)
                  
                  
def qml_compute_gs(H):
    '''
    Compute the Ground State given the Hamiltonian
    '''
    return np.min(qml.eigvals(H))

                  
#         ,d8       
#       ,d888       
#     ,d8" 88       
#   ,d8"   88       
# ,d8"     88       
# 8888888888888     
#          88  888  
#          88  888  Circuits

def vqe_ising_chain_circuit(param, N):
    for spin in range(N):
        qml.RY(param[spin], wires = spin)
    
    # Apply entanglement to the neighbouring spins
    for spin in range(N-1):
        qml.CNOT(wires = [spin, spin+1])
    
    qml.Barrier()
    
    # Apply Y, Z, Y rotations to cover the entire bloch sphere
    for spin in range(N):
        qml.RY(param[N   + spin], wires = spin)
        #qml.RZ(param[2*N + spin], wires = spin)
        #qml.RY(param[3*N + spin], wires = spin)
                  
                  
# 8888888888        
# 88                
# 88  ____          
# 88a8PPPP8b,       
# PP"     `8b       
#          d8       
# Y8a     a8P  888  
#  "Y88888P"   888  Training

def train_vqe_ising(step_size, l_steps, epochs, N, J, dev, circuit = False, plots = False):
    '''
    step_size = Step size of the optimizer
    epochs    = Max epochs for each lambda (magnetic field)
    lams      = Array of intensities of magnetic field
    N         = Number of spins of the system
    '''
    
    display(Markdown('***Parameters:***'))
    print('step_size = {0} (Step size of the optimizer)'.format(step_size))
    print('epochs    = {0} (Max epochs for each lambda)'.format(epochs))
    print('N         = {0} (Number of spins of the system)'.format(N))
    
    lams = np.linspace(0,2*J,l_steps)
    
    vqe_e = []
    errs = np.zeros((epochs, len(lams)) )
    
    @qml.qnode(dev)
    def cost_fn(param):
        vqe_ising_chain_circuit(param, N)
        # return <psi|H|psi>
        return qml.expval(H)
    
    if circuit:
        # Display the circuit
        display(Markdown('***Circuit:***'))
        
        H = qml_build_H(N, 0, 0)        
        drawer = qml.draw(cost_fn)
        print(drawer(np.array([0]*(4*N))))
    
    thetas = np.array([0]*(4*N), requires_grad = True) # Prepare initial state
    thetas_arr = []
    ys = [] # Labels of the phase
    for i, l in enumerate(tqdm(lams)):
        opt = qml.GradientDescentOptimizer(stepsize=step_size)
        H = qml_build_H(N, float(l), J)
        
        # Compute GS-energy by taking the lowest Eigenvalue from the known matrix
        GS_H = qml_compute_gs(H)
        
        for epoch in range(epochs):
            # Compute <H>, then update thetas
            thetas, prev_energy = opt.step_and_cost(cost_fn, thetas)
            # Store the MSE of the current (epoch, lambda)-pair
            errs[epoch, i] = (prev_energy - GS_H)**2
        
        vqe_e.append(cost_fn(thetas))
        thetas_arr.append(thetas)
        ys.append(0) if l <= J else ys.append(1)
        
    if plots:
        fig, ax = plt.subplots(2, 1, figsize=(10,10))

        true_e = []
        for l in lams:
            H = qml_build_H(N, float(l), J)
            true_e.append(qml_compute_gs(H))
            
        ax[0].plot(lams, true_e, '--', label='True', color='red', lw = 2)
        ax[0].plot(lams, vqe_e, '.', label='VQE', color='green', lw = 2)
        ax[0].plot(lams, vqe_e, color='green', lw = 2, alpha=0.6)
        ax[0].grid(True)
        ax[0].set_title('Ground States of Ising Hamiltonian ({0}-spins), J = {1}'.format(N,J))
        ax[0].set_xlabel(r'$\lambda$')
        ax[0].set_ylabel(r'$E(\lambda)$')
        ax[0].legend()

        ax[1].plot(range(epochs), np.mean(errs, axis=1), '.', color='orange', ms = 7 )
        ax[1].plot(range(epochs), np.mean(errs, axis=1), color='orange', alpha=0.4)
        ax[1].set_title('Convergence of VQE')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('MSE')
        ax[1].grid(True)
        ax[1].axhline(y=0, color='r', linestyle='--')
        
        plt.show()
        
        plt.title('J = {0}'.format(J)) 

        pm = plt.imshow(np.abs(thetas_arr))

        for i in range(np.shape(thetas_arr)[1]):
            plt.axvline(x=i - .5, color = 'black')

        plt.ylabel(r'$\lambda$')

        plt.colorbar(pm, shrink=0.3, aspect=20)
        plt.tight_layout()
        plt.show()  
        
    return vqe_e, errs, thetas_arr, ys                  
                  
                  
#   ad8888ba,       
#  8P'    "Y8       
# d8                
# 88,dd888bb,       
# 88P'    `8b       
# 88       d8       
# 88a     a8P  888  
#  "Y88888P"   888  Display

def show_params_ising_chain(thetas_arr):
    '''
    thetas_arr: array of the following structure:
                    list( (J_value, angles of the circuit (vqe_ising_chain_circuit) ) )
    '''
    for J in range(len(thetas_arr)):
        plt.title('J = {0}'.format(thetas_arr[J][0])) 

        pm = plt.imshow(np.abs(thetas_arr[J][1]))

        for i in range(np.shape(thetas_arr[J][1])[1]):
            plt.axvline(x=i - .5, color = 'black')

        plt.ylabel(r'$\lambda$')

        plt.colorbar(pm, shrink=0.3, aspect=20)
        plt.tight_layout()
        plt.show()                  
                  
                  
                  


                    