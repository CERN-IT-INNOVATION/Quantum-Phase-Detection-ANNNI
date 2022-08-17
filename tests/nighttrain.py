"""
This script saves a trained VQE 
To run it:
    ./getvqeparams_ANNNI.py N side epochs ring
    ring = 0 False
    ring = 1 True
"""

# Other
import sys
sys.path.insert(0, '../')

# VQE - Modules
import PhaseEstimation.vqe as vqe
import PhaseEstimation.annni_model as annni
import PhaseEstimation.hamiltonians as ham

# Get variables from command line arguments:
Ns = [12]
side = 100
for N in Ns:
    print(N)
    # Initialize Hamiltonian
    #Hs = ham.hamiltonian(annni.build_Hs, N = N, n_states = side, ring = False)
    # Initialize VQE
    myvqe = vqe.load_vqe('./N'+str(N)+'n'+str(side))

    # Train VQE recycling parameters 
    #myvqe.train(0.3, 250, recycle = True, epochs_batch_size = 250, batch_size = 1)
    #myvqe.save('./N'+str(N)+'n'+str(side))
    myvqe.train_refine(.2,  1000, 0.005, assist = True)
    myvqe.save('./N'+str(N)+'n'+str(side))
    myvqe.train_refine(.1,  1000, 0.005, assist = True)
    myvqe.save('./N'+str(N)+'n'+str(side))
    
    del(myvqe)
