[![Made at CERN!](https://img.shields.io/badge/CERN-CERN%20openlab-blue)](https://openlab.cern/) 
[![DOI](https://zenodo.org/badge/478113360.svg)](https://zenodo.org/badge/latestdoi/478113360)
[![DOI:10.48550/arXiv.2208.08748](http://img.shields.io/badge/DOI-10.48550/arXiv.2208.08748-B31B1B.svg)](https://doi.org/10.48550/arXiv.2208.08748)

Read the Docs Documentation: https://cern-qpd-annni.readthedocs.io/en/latest/

---

# Quantum phase detection generalisation from marginal quantum neural network models

Package build on Pennylane for the Quantum Phase Detection of the ANNNI Model through Quantum Convolutional Neural Networks and Quantum Autoencoder

## How to install

### Create and activate the environment

1. ```$ python3 -m venv <env-name>```
2. ```$ source <env_name>/bin/activate```

### Clone and move to Project folder

3. ```git clone https://github.com/CERN-IT-INNOVATION/Quantum-Phase-Detection-ANNNI.git```
4. ```cd Quantum-Phase-Detection-ANNNI```

### Install required packages

5. ```pip install ./```

### (Optional) To run on GPU

6. ```pip install --upgrade pip```
7. ```pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```

## Examples

The [/notebooks](notebooks) folder contains many examples for all the use-cases as Jupyter Notebooks

---

## How to cite

If you used this package for your research, please cite:

```text
@article{PhysRevB.107.L081105,
  title = {Quantum phase detection generalization from marginal quantum neural network models},
  author = {Monaco, Saverio and Kiss, Oriel and Mandarino, Antonio and Vallecorsa, Sofia and Grossi, Michele},
  journal = {Phys. Rev. B},
  volume = {107},
  issue = {8},
  pages = {L081105},
  numpages = {6},
  year = {2023},
  month = {Feb},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.107.L081105},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.107.L081105}
}
```
