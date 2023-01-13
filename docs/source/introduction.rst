Quantum phase detection generalisation from marginal quantum neural network models
==================================================================================

Package build on Pennylane for the Quantum Phase Detection of the ANNNI Model through Quantum Convolutional Neural Networks and Quantum Autoencoder

How to install:
---------------
Create and activate the environment

.. code-block:: bash

   $ python3 -m venv <env-name>
   $ source <env_name>/bin/activate

Clone and move to Project folder

.. code-block:: bash

   git clone https://github.com/CERN-IT-INNOVATION/Quantum-Phase-Detection-ANNNI.git
   cd Quantum-Phase-Detection-ANNNI

Install required packages

.. code-block:: bash

   pip install .

(Optional) To run on GPU

.. code-block:: bash

   pip install --upgrade pip
   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Examples
--------
The `/notebooks <https://github.com/CERN-IT-INNOVATION/Quantum-Phase-Detection-ANNNI/tree/main/notebooks>`_ folder contains many examples for all the use-cases as Jupyter Notebooks


How to cite
-----------
If you used this package for your research, please cite


Bibtex:

.. code-block:: latex

   @article{monaco2022quantum,
     title={Quantum phase detection generalisation from marginal quantum neural network models},
     author={Monaco, Saverio and Kiss, Oriel and Mandarino, Antonio and Vallecorsa, Sofia and Grossi, Michele},
     journal={arXiv preprint arXiv:2208.08748},
     year={2022}
    }
