"""Module to deal with the ANNNI Hamiltonians"""
import pennylane.numpy as np
from jax import jit 
import jax.numpy as jnp
import pennylane as qml
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def paraanti(x: np.ndarray) -> np.ndarray:
    r"""
    Transition Line: Paramagnetic-Antiphase [0.5 < x < 1.0].
    The function calculates the transition line defined by:

        .. math::
            y(x) = 1.05 \sqrt{(x - 0.5)*(x - 0.1)}

    Parameters
    ----------
    x : np.ndarray
        Input value(s) where the transition line is calculated.

    Returns
    -------
    np.ndarray
        Output array representing the transition line values.
    """

    y = 1.05 * np.sqrt(np.maximum((x - 0.5) * (x - 0.1), 0))
    y[x < .5] = 0
    return y

def paraferro(x: np.ndarray) -> np.ndarray:
    r"""
    Transition Line: Paramagnetic-Ferromagnetic [0.0 < x < 0.5].

    The function calculates the transition line defined by:

    .. math::
        y(x) = \frac{1 - x}{x} \left(1 - \sqrt{\frac{1 - 3x + 4x^2}{1 - x}}\right)

    Parameters
    ----------
    x : np.ndarray
        Input value(s) where the transition line is calculated. Values greater than 0.5 are set to NaN.

    Returns
    -------
    np.ndarray
        Output array representing the transition line values.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_x = np.where(x == 0, np.nan, x)
        y = ((1 - safe_x) / safe_x) * (1 - np.sqrt((1 - 3 * safe_x + 4 * safe_x * safe_x) / (1 - safe_x)))
        y[np.isnan(y)] = 1  # Replace NaN with zero or another desired value
        y[safe_x > .5] = 0 # Set to 0 the phase transition line out of his domain
    return y

def b1(x: np.ndarray) -> np.ndarray:
    r"""
    Floating phase Transition Line [0.5 < x < 1.0].

    The function calculates the transition line defined by:

    .. math::
        y(x) = 1.05(x - 0.5)

    Parameters
    ----------
    x : np.ndarray
        Input value(s) where the transition line is calculated.

    Returns
    -------
    np.ndarray
        Output array representing the transition line values.
    """
    y = 1.05 * (x - 0.5)
    y[x < .5] = 0
    return y

def peshel_emery(x: np.ndarray, y_cap : float = 2) -> np.ndarray:
    r"""
    Peshel-Emery Line.

    The function calculates the line defined by:

    .. math::
        y(x) = \frac{1}{4x} - x

    Values greater than y_cap are capped at y_cap.

    Parameters
    ----------
    x : np.ndarray
        Input value(s) where the line is calculated.
    y_cap : float
        Cap value

    Returns
    -------
    np.ndarray
        Output array representing the line values.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        y = (1 / (4 * x)) - x
        y[y>y_cap] = y_cap  # Handle large results
        y[x > .5]  = 0  # Set to 0 the phase transition line out of his domain
        y = np.where(np.isfinite(y), y, y_cap)  # Handle infinite results
    return y

j_linalgeigh = jit(jnp.linalg.eigh)

class Annni():
    def __init__(self, n_spin : int, k: float, h: float, n_spin_simulable : int = 12, ring : bool = False):
        """
        Initialize the Annni model with given parameters.

        Parameters
        ----------
        n_spin : int
            Number of spins of the system
        k : float
            The parameter k, must be within the range [0, 1].
        h : float
            The parameter h, must be within the range [0, 2].
        n_spin_simulable : int
            Maximum number of spins of the system for which energy will be obtained 
            through diagonalization
        ring : bool 
            If True -> periodic boundary conditions
        """

        if not (0 <= h <= 2):
            raise ValueError(f"Parameter 'h' must be between 0 and 2. Got: {h}")
        if not (0 <= k <= 1):
            raise ValueError(f"Parameter 'k' must be between 0 and 1. Got: {k}")
        
        self.n_spin = n_spin
        self.h = h
        self.k = k
        self.ring = ring
        self.H = self.get_H()
        self.phase = self.get_phase()
        self.analytical = self.is_analytical()
        
        # If the number of spin is simulable (meaning that we can actually
        # diagonalize the state's hamiltonian in a reasonable time)
        # compute the ground state energy and wavefunction
        # energy can be used to estimate how well the VQE algorithm is performing
        # wavefunction can be used as input bypassing the VQE procedure
        if self.n_spin <= n_spin_simulable:
            self.energy, self.psi = self.get_energy_psi()

    def get_H(self):
        """Construction function the ANNNI Hamiltonian (J=1)"""

        # Interaction between spins (neighbouring):
        H = -1 * (qml.PauliX(0) @ qml.PauliX(1))
        for i in range(1, self.n_spin - 1):
            H = H  - (qml.PauliX(i) @ qml.PauliX(i + 1))

        # Interaction between spins (next-neighbouring):
        for i in range(0, self.n_spin - 2):
            H = H + self.k * (qml.PauliX(i) @ qml.PauliX(i + 2))

        # Interaction of the spins with the magnetic field
        for i in range(0, self.n_spin):
            H = H - self.h * qml.PauliZ(i)

        return H
    
    def get_phase(self):
        # 0 -> Ferromagnetic
        # 1 -> Paramagnetic
        # 2 -> Antiphase

        # Left side, it can be either Ferromagnetic of paramagnetic
        if self.k <= .5:
            return 0 if self.h <= paraferro(np.array([self.k]))[0] else 1
        # Right side, it can be either Paramagnetic or Antiphase
        else:
            return 2 if self.h <= paraanti(np.array([self.k]))[0] else 1

    def is_analytical(self):
        # If the point in the phase space lies in one of the two axes, the 
        # point is indeed analytical, meaning that it is possible to 
        # get its phase without any approximate methods
        return self.h == 0 or self.k == 0

    def get_energy_psi(self):
        # Get the matrix form of the Hamiltonian and ensure it's real
        mat_H = np.real(qml.matrix(self.H)).astype(np.single)

        # Compute sorted eigenvalues and eigenvectors using jitted function
        eigvals, psi = j_linalgeigh(mat_H)  # psi contains eigenvectors as columns

        # Get the ground-state energy (smallest eigenvalue)
        en0 = eigvals[0]

        # Get the ground-state wavefunction (first eigenvector)
        psi0 = psi[:, 0]

        return en0, psi0
            
            
def set_layout(title='Phase-Space'):
    # X values for different phase boundaries
    xl = np.linspace(0, 0.5, 50)
    xpe = np.linspace(0.12, 0.5, 50)
    xr = np.linspace(0.5, 1, 50)

    # Plot the transition lines
    plt.plot(xpe, peshel_emery(xpe, y_cap=2), ls = '--', color='black')
    plt.plot(xr, paraanti(xr), color='black')
    plt.plot(xl, paraferro(xl), color='black')
    plt.plot(xr, b1(xr), color='black', ls='--')
    plt.plot([], [], color='black', label='Transition Lines')

    plt.legend()
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("h")

    # Explicitly set the correct y-limits
    plt.ylim(0, 2)

def show_phases(side : int = 40):
    p_k = np.linspace(0, 1, side)
    p_h = np.linspace(0, 2, side)
    
    img = np.zeros((side, side))
    for x, k in enumerate(p_k):
        for y, h in enumerate(p_h):
            phase = Annni(2, k=k, h=h).phase
            if phase == 2 and (h > b1(np.array([k]))):
               phase = 3
            img[x,y] = phase

    # Create a custom colormap
    plt.figure(figsize=(4,4))
    colors = ['#4d94d7', '#68d16c', '#ffd34d', '#ffb54d']
    p_phase = ["Ferromagnetic", "Paramagnetic", "Antiphase", "Floatingphase"]
    for color, phase in zip(colors, p_phase):
        plt.scatter([], [], color=color, label=phase)
    cmap = ListedColormap(colors)
    plt.imshow(np.flip(np.rot90(img, k=-1),axis=1), cmap = cmap, aspect="auto", origin="lower", extent=[0, 1, 0, 2])
    set_layout()