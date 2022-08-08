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

# Load VQE
if sys.argv[4]:
	myvqe = vqe.load_vqe('./data/vqes/ANNNI/N'+str(N)+'n'+str(side)+'ring.pkl')
else:
	myvqe = vqe.load_vqe('./data/vqes/ANNNI/N'+str(N)+'n'+str(side)+'.pkl')

# Train VQE recycling parameters 
myvqe.train_excited(0.3, epochs, 5)
# Further train low accuracy points
myvqe.train_refine_excited(0.1, 2*epochs, 0.01, 5)

# Save the VQE
if sys.argv[4] == 1:
	myvqe.save('./N'+str(N)+'n'+str(side)+'ring.pkl')
else:
	myvqe.save('./N'+str(N)+'n'+str(side)+'.pkl')
