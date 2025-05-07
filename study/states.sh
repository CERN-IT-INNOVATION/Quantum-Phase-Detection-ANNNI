#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --chdir=/home/samonaco/Quantum-Phase-Detection-ANNNI/study 
#SBATCH --job-name=ANNNI_dmrg                    
#SBATCH --output=ANNNI_dmrg.out
#SBATCH --error=ANNNI_dmrg.err
#SBATCH --mail-type=ALL

echo "Activating Environment"
source /home/samonaco/Quantum-Phase-Detection-ANNNI/env/bin/activate

echo "Running script"
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/exact_states.py -L 6  -side 51        -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 6  -side 51 -chi 2 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 6  -side 51 -chi 4 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 6  -side 51 -chi 8 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/

python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/exact_states.py -L 9  -side 51        -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 9  -side 51 -chi 2 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 9  -side 51 -chi 4 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 9  -side 51 -chi 8 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/

python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/exact_states.py -L 12 -side 51        -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 12 -side 51 -chi 2 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 12 -side 51 -chi 4 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 12 -side 51 -chi 8 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/

python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/exact_states.py -L 15 -side 51        -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 15 -side 51 -chi 2 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 15 -side 51 -chi 4 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/
python3 /home/samonaco/Quantum-Phase-Detection-ANNNI/study/dmrg_states.py  -L 15 -side 51 -chi 8 -path /home/samonaco/Quantum-Phase-Detection-ANNNI/study/mps/