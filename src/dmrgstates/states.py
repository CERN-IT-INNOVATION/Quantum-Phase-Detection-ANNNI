import hamiltonians as H
import annni_model  as ANNNI
import general      as qmlgen
import numpy as np

def get_indexes_axes(n_hs, n_kappas):
    x_indexes = np.arange(0, n_kappas*n_hs+1, n_hs)
    y_indexes = np.arange(0, n_hs,            1)

    return x_indexes, y_indexes

def get_labels(h, k):
    if k == 0:
        # Added this case because of 0 encountering in division
        if h <= 1:
            return 0, 0
        else:
            return 1, 1
    elif k > -.5:
        # Left side (yes it is the left side, the phase plot is flipped due to x axis being from 0 to - kappa_max)
        if h <= qmlgen.paraferro(-k):
            return 0, 0
        else:
            return 1, 1
    else:
        # Right side
        if h <= qmlgen.paraanti(-k):
            if h <= qmlgen.b1(-k):
                return 2, 2
            else:
                return 2, 3
        else:
            return 1, 1

class exact_states:
    def __init__(self, N : int = 6, n_hs : int = 10, n_kappas : int = 10, h_max: float = 2, kappa_max: float = 1):
        # Input parameters
        self.N, self.n_hs, self.n_kappas, self.h_max, self.kappa_max = N, n_hs, n_kappas, h_max, kappa_max

        # (ANNNI) Hamiltonian class
        self.Hs          = H.hamiltonian(ANNNI.build_Hs, N = N, n_hs = n_hs, n_kappas = n_kappas, kappa_max = kappa_max, h_max = h_max, ring = False)
        # Energies and (exact) wavefuctions
        self.E, self.PSI = H.get_e_psi(self.Hs, 0)
        # h and kappas values
        self.H_params    = self.Hs.model_params[:,1:]

        # LABELS
        # -> Y3 : Labels for the 3 phases
        # -> Y4 : Labels for the 3 phases + 1 (floating)    
        Y3, Y4 = [], [] 
        for h, k in self.H_params:
            y3, y4 = get_labels(h, k)
            Y3.append(y3)
            Y4.append(y4)
            
        self.Y3, self.Y4 = np.array(Y3), np.array(Y4)

    def __repr__(self):
        self.Hs.show_phasesplot()
        return ''
    