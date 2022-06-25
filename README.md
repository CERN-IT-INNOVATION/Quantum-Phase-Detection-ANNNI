# Phase Estimation through VQE+QCNN
## Activate the environment:
1. ```$ sudo apt install virtualenv```
2. ```$ virtualenv <env_name>```
3. ```$ source <env_name>/bin/activate```
4. ```(<env_name>)$ pip install -r path/to/requirements.txt```

---
## Resources:
### Quantum Programming
- qiskit textbook https://qiskit.org/textbook/preface.html
- pennylane (faster simulations than qiskit) https://pennylane.ai/qml/demonstrations.html

### VQE - ground state preparation
- original paper https://arxiv.org/abs/1304.3061
- recent review https://arxiv.org/abs/2103.08505
- tutorial https://qiskit.org/textbook/ch-applications/vqe-molecules.html, https://pennylane.ai/qml/demos/tutorial_vqe.html, https://qiskit.org/documentation/nature/tutorials/index.html

### Classification on Quantum Ground-State
- QCNN (architecture we are interested in and original paper) https://arxiv.org/abs/1810.03787
- easy application on a Ising chain (i would recommend we start reproducing these results) https://arxiv.org/abs/1906.10155
- multi-class https://arxiv.org/abs/2110.08386 
- nice multi-class usecase (but the main text is not that relevant for us)  https://arxiv.org/abs/2111.05292
- quantum advantage by learning from experiment (for general interest, theoretical paper) https://arxiv.org/abs/2112.00778 
---
