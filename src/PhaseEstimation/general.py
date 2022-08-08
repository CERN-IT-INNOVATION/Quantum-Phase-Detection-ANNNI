"""Module for generic functions for other modules"""
import pennylane as qml
from pennylane import numpy as np
import jax
import jax.numpy as jnp
from jax import jit


def linalgeigh(mat_H):
    """
    Apply jax np.linalg.eigh on a matrix,
    to be jitted
    
    Parameters
    ----------
    params : np.ndarray
        Input matrix to apply np.linalg.eigh

    Returns
    -------
    np.ndarray
        Array of eigenvalues (not sorted)
    np.ndarray
        Array of relatives eigenvectors
    """
    # Compute eigenvalues and eigenvectors
    eigval, eigvec = jnp.linalg.eigh(mat_H)
    
    return eigval, eigvec

j_linalgeigh = jax.jit(linalgeigh)

def geteigvals(qml_H, states):
    """
    Function for getting the energy values of an Ising Hamiltonian
    using the jitted jnp.linalg.eigh function
        
    Parameters
    ----------
    qml_H : pennylane.ops.qubit.hamiltonian.Hamiltonian
        Pennylane Hamiltonian of the Ising Model
    states : list
        List of energy levels desired

    Returns
    -------
    list
        List of energy values
    """
    
    # Get the matricial for of the hamiltonian
    # and convert it to float32 
    # This type of hamiltonians are always real
    mat_H = np.real(qml.matrix(qml_H)).astype(np.single)
    
    # Compute sorted eigenvalues with jitted function
    eigvals = jnp.sort(j_linalgeigh(mat_H)[0])
    
    return [eigvals[k] for k in states]



def get_H_eigval_eigvec(qml_H, en_lvl):
    """
    Function for getting the energy value and state of an Ising Hamiltonian
    using the jitted jnp.linalg.eigh function
        
    Parameters
    ----------
    qml_H : pennylane.ops.qubit.hamiltonian.Hamiltonian
        Pennylane Hamiltonian of the Ising Model
    en_lvl : int
        Energy level desired

    Returns
    -------
    np.ndarray
        Matricial encoding of the Hamiltonian
    float
        Value of the energy level
    np.ndarray
        Eigenstate of the energy level
    """
    
    # Get the matricial for of the hamiltonian
    # and convert it to float32 
    # This type of hamiltonians are always real
    mat_H = np.real(qml.matrix(qml_H)).astype(np.single)
    
    # Compute sorted eigenvalues with jitted function
    eigvals, eigvecs = j_linalgeigh(mat_H)
    
    psi = eigvecs[:,jnp.argsort(eigvals)[en_lvl]]
    en = jnp.sort(eigvals)[en_lvl]
    
    return mat_H, en, psi

def psi_outer(psi):
    """
    Apply the outer product between two vectors
    (mainly used for imposing orthogonality in VQD)
    to be jitted
    
    Parameters
    ----------
    psi : np.ndarray
        Input vector (eigenstate)

    Returns
    -------
    np.ndarray
        Outer product |psi><psi|
    """
    
    return jnp.outer(jnp.conj(psi), psi)

j_psi_outer = jax.jit(psi_outer)
jv_psi_outer = jax.jit(jax.vmap(psi_outer))

def get_VQD_params(qml_H, beta):
    """
    Function for getting all the training parameter for the VQD
    algorithm for finding the first excited state
        
    Parameters
    ----------
    qml_H : pennylane.ops.qubit.hamiltonian.Hamiltonian
        Pennylane Hamiltonian of the Ising Model
        
    Returns
    -------
    np.ndarray
        Matricial encoding of the Hamiltonian
    np.ndarray
        Effective Hamiltonian of VQD algorithm
    float
        Excited energy value
    """
    
    # Get the matricial for of the hamiltonian
    # and convert it to float32 
    # This type of hamiltonians are always real
    mat_H = np.real(qml.matrix(qml_H)).astype(np.single)
    
    # Compute sorted eigenvalues with jitted function
    eigvals, eigvecs = j_linalgeigh(mat_H)
    
    psi0 = eigvecs[:,jnp.argsort(eigvals)[0]]
    en_ex = jnp.sort(eigvals)[1]
    
    return jnp.array([mat_H]), jnp.array([mat_H + beta * j_psi_outer(psi0)]), en_ex

def get_neighbours(vqeclass, idx):
    """
    Function for getting the neighbouring indexes
    (up, down, left, right) of a given state (K, L)
    in the ANNNI model.
    
    Examples
    --------
    
    Indexes:
    +--------------+
    | 4  9  14  19 |
    | 3  8  13  18 |
    | 2  7  12  17 |
    | 1  6  11  16 |
    | 0  5  10  15 |
    +--------------+
    
    >>> get_neighbours(vqeclass, 0)
    array([1, 5])
    >>> get_neighbours(vqeclass, 12)
    array([7, 13, 17, 11])
    
    Parameters
    ----------
    vqeclass : class
        Class of the VQE, used to get the size of the side and the recycle rule
    idx : int
        Index of the desired state

    Returns
    -------
    np.ndarray
        Neighbouring indexes
    """
    
    side = int(np.sqrt(vqeclass.n_states))
    
    neighbours = np.array([idx + 1, idx - 1, idx + side, idx - side])
    neighbours = np.delete(neighbours, np.logical_not(np.isin(neighbours, vqeclass.Hs.recycle_rule)) )


    if (idx + 1) % side == 0 and idx != vqeclass.n_states - 1:
        neighbours = np.delete(neighbours, 0)
    if (idx    ) % side == 0 and idx != 0:
        neighbours = np.delete(neighbours, 1)
        
    return neighbours

def findC(*vqes, width, info = False, plot = False):
    """
    Find the critical line in L for K in [0,.5]
    
    By fixing K (namely J1 and J2) and computing the energy gaps between the
    first excited state and ground state for L in [0,2] and for various Ns
    the critical lines of each N-system will cross at a point near the phase transition.
    This leads to accurate results for L in [0,.5]
    
    Parameters
    ----------
    *vqes : *list
        *list of trained (ground states and excited) vqes
    width : float
        width for the range of the crossing finding algorithm
    """
    
    
    def find_cross(vqe1, vqe2, idxs):
        delta_e1 = (vqe1.vqe_e1[idxs] - vqe1.vqe_e0[idxs])*vqe1.Hs.N 
        delta_e2 = (vqe2.vqe_e1[idxs] - vqe2.vqe_e0[idxs])*vqe2.Hs.N 
        
        dist = delta_e1 - delta_e2
        
        signchanges = (np.diff(np.sign(dist)) != 0)*1
        
        return np.where(signchanges == 1)[0]
        
    n_states = [vqe.Hs.n_states for vqe in vqes]
    
    if len(np.unique(n_states)) > 1:
        print('Invalid VQEs, different sides not supported')
    n_states = n_states[0]
    side     = int(np.sqrt(n_states))
           
    if info:
        print('Number of VQEs: {0}'.format(len(vqes)) )
        print('Spins of VQEs: ', [vqe.Hs.N for vqe in vqes] )
        print('Side of VQEs:  ', side)
        
    j2j1 = np.unique(vqes[0].Hs.model_params[:,2])
        
    getrange = lambda center, width: [center-width/8, center + width/2]
    center = 0
    centers = []
    
    for i, L in enumerate(j2j1):
        if L > -.5:
            idxs = np.arange(n_states)[vqes[0].Hs.model_params[:,2] == L]

            crosses = []
            for k, vqe1 in enumerate(vqes):
                for vqe2 in vqes[k+1:]:
                    vqepaircrosses = find_cross(vqe1, vqe2, idxs) 
                    for vqepaircross in vqepaircrosses:
                        crosses.append(vqepaircross)

            crosses = np.array(crosses)
            interval = getrange(center, width)

            cross_center = 2*np.mean(crosses[np.logical_and(2*crosses/side > interval[0], 2*crosses/side <  interval[1] )])/side
            center = cross_center
            centers.append(center)
            
            if i%10 == 0:
                if plot:
                    for vqe in vqes:
                        delta_es = (vqe.vqe_e1[idxs] - vqe.vqe_e0[idxs])*vqe.Hs.N
                        plt.plot(np.linspace(0,2,side), delta_es, label = vqe.Hs.N)

                        for cross in crosses:
                            plt.axvline(x=2*cross/side, alpha = .1)

                    plt.text(1.3, 20, r'$J_2/\,J_1=$'+str(np.round(L,2)))
                    plt.axvline(x=center, color = 'red')
                    plt.legend()
                    plt.show()
            
    return np.array(centers)