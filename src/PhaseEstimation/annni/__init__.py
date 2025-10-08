"""Module to deal with the ANNNI Hamiltonians"""

import jax.numpy as jnp
import tqdm

from .hamiltonian import get_energy_psi, get_H
from .lines import get_phase, is_analytical


class Annni:
    def __init__(
        self,
        num_spins: int,
        k: float,
        h: float,
        is_simulable: bool = False,
        ring: bool = False,
        usejax: bool = False,
    ):
        if not (0 <= h <= 2):
            raise ValueError(f"Parameter 'h' must be between 0 and 2. Got: {h}")
        if not (0 <= k <= 1):
            raise ValueError(f"Parameter 'k' must be between 0 and 1. Got: {k}")

        self.num_spins = num_spins
        self.h = h
        self.k = k
        self.ring = ring
        self.H = get_H(num_spins, k, h, ring)
        self.phase = get_phase(k, h)
        self.analytical = is_analytical(k, h)

        # If the number of spin is simulable (meaning that we can actually
        # diagonalize the state's hamiltonian in a reasonable time)
        # compute the ground state energy and wavefunction
        # energy can be used to estimate how well the VQE algorithm is performing
        # wavefunction can be used as input bypassing the VQE procedure
        if is_simulable:
            self.energy, self.psi = get_energy_psi(
                num_spins, k, h, ring=ring, usejax=usejax
            )
