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
N = int(sys.argv[1])
side = int(sys.argv[2])
epochs = int(sys.argv[3])

# Initialize Hamiltonian
Hs = ham.hamiltonian(annni.build_Hs, N = N, n_states = side, ring = bool(sys.argv[4]) )
# Initialize VQE
myvqe = vqe.vqe(Hs, vqe.circuit_ising)

# Train VQE recycling parameters 
myvqe.train(0.3, epochs, circuit = False, recycle = True, epochs_batch_size = epochs, batch_size = 1)
# Further train low accuracy points
myvqe.train_refine(0.1, 2*epochs, 0.01)

# Save the VQE
if sys.argv[4] == 1:
	myvqe.save('./N'+str(N)+'n'+str(side)+'ring.pkl')
else:
	myvqe.save('./N'+str(N)+'n'+str(side)+'.pkl')
