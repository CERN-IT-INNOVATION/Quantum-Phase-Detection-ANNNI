""" This module implements the base function for the VQE model"""
import pennylane as qml
import pennylane.numpy as np
from PhaseEstimation import annni, circuits
import tqdm.notebook as tqdm
from typing import Callable

class Vqe:
    def __init__(self, n_qubit : int, side : int, ansatz: Callable = circuits.ID9, **kwargs):
        """
        VQE Class for the ANNNI model
        
        Parameters
        ----------
        n_qubit : int
            Number of qubits
        side : int
            Level of discretization of the phase space
        ansatz : Callable
            Pennylane Ansatz
        """
        self.n_qubit = n_qubit
        self.side = side
        self.ansatz = ansatz
        self.kwargs = kwargs

        self.p_h = np.linspace(0,2,self.side)
        self.p_k = np.linspace(0,1,self.side)
        self.device = qml.device("default.qubit", wires=self.n_qubit, shots=None)

        self.dict_p_p : dict = {}
        self.dict_Y : dict = {}
        self.dict_E   : dict = {}
        self.dict_analytical : dict = {}
                
        self.n_p = self.ansatz(self.n_qubit, np.arange(10000), **self.kwargs)
        self.q_circuit = qml.QNode(self.circuit, device=self.device) 

        self.recycle_rule = []
        up = False
        for k in self.p_k:
            self.recycle_rule += [(float(k), float(h)) for h in self.p_h[::2*int(up) - 1]]
            up = not up
        
    def circuit(self, p_v, H):
        self.ansatz(self.n_qubit, p_v, **self.kwargs)
        return qml.expval(H)
        
    def __repr__(self):
        repr_str  = "VQE Class:"
        repr_str += f"\n  N : {self.n_qubit}"
        repr_str += f"\n  side : {self.side}"
        repr_str +=  "\n  Circuit:\n"
        repr_str += qml.draw(self.q_circuit)(np.arange(self.n_p), annni.Annni(self.n_qubit, 0, 0).H)
        
        return repr_str
    
    def train_site(self, n_epoch : int, lr : float, p_v0 : None | np.ndarray, H, E_true : None | float = None, atol : float = 1e-4, bar_position : int = 0):
        if p_v0 is None:
            p_v = np.random.uniform(-.3*np.pi, +3*np.pi, size=self.n_p, requires_grad = True)
        else:
            p_v = p_v0

        @qml.qnode(self.device, interface="autograd")
        def cost_func(p_v):
            self.ansatz(self.n_qubit, p_v, **self.kwargs)
            return qml.expval(H)
        
        opt = qml.AdamOptimizer(stepsize=lr)
        progress = tqdm.tqdm(range(1, n_epoch + 1), position=bar_position, leave=False)

        for epoch in range(1, n_epoch + 1):
            p_v, E = opt.step_and_cost(cost_func, p_v)
            if E_true is not None:
                E_diff = abs(E - E_true)
                progress.set_description(f"Abs. Error: {E_diff:.4f}")
                if E_diff <= atol:
                    progress.n = len(progress)  # Set progress to 100%
                    break
            progress.update(1)  # Refresh the bar
        progress.close()
        return p_v, E
    
    def train_all(self, n_epoch : int, lr : float, atol : float = 1e-4, epoch_factor : float = 10, lr_factor : float = 10, p_v0 : None | np.ndarray = None):
        """
        Training function with recycle rule (use trained parameters of a neighbor state)
        
        Parameters
        ----------
        n_epoch : int
            Number of epochs
        lr : float
            Learning rate of the optimizer
        atol : float
            Stop training if the absolute difference between the VQE energy and true energy is lower thana atol
        epoch_factor : float
            Divide the training by epoch_factor for training that use the recycle rule
        lr_factor : float
            Divide the learning rate by lr_factor for training that use the recycle rule
        p_v0 : np.ndarray
            Initial parameters
        """
        progress = tqdm.tqdm(range(len(self.p_k)*len(self.p_h)), position = 0, leave = True)        

        p_v_prev = None if p_v0 is None else p_v0
        for i, (k, h) in enumerate(self.recycle_rule):
            H = annni.Annni(self.n_qubit, k, h)
            progress.set_description(f"k: {k:.3f}, h: {h:.3f}")
            p_v_prev, E_prev = self.train_site(n_epoch, lr, p_v_prev, H.H, H.energy, atol, 1)
            self.dict_p_p[(k, h)] = p_v_prev
            self.dict_E[(k, h)] = E_prev
            self.dict_Y[(k, h)] = H.phase
            self.dict_analytical[(k, h)] = H.analytical

            if i == 0:
                lr = lr/lr_factor
                n_epoch = int(n_epoch/epoch_factor)
            progress.update(1)
                
        return
