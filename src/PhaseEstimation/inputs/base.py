import abc

import jax.numpy as jnp
import numpy as np

from ..annni.lines import get_phase, is_analytical


class Input(abc.ABC):
    """Abstract base class for quantum circuit blocks."""

    def __init__(self, num_qubits: int, ks: np.ndarray, hs: np.ndarray):
        self.num_qubits = num_qubits
        self.ks = np.sort(np.unique(ks))
        self.hs = np.sort(np.unique(hs))
        assert len(self.ks) == len(self.hs), "ks and hs must have the same length"

        self.side = len(self.ks)

    @abc.abstractmethod
    def ansatz(self, input_obj):
        """Subclasses must implement this to define their circuit structure."""
        pass

    @abc.abstractmethod
    def make(self, *args, **kwargs):
        """Make dataset"""
        pass

    def load_single(self, k, h):
        """Get the state and label"""
        return self.input_dict[(k, h)], jnp.eye(4)[get_phase(k, h)]

    def load_all(self):
        analytical_mask = []
        index_map = []

        states, labels = [], []

        i = 0
        for k in self.ks:
            for h in self.hs:
                state, label = self.load_single(k, h)
                states.append(state)
                labels.append(label)

                if is_analytical(k, h):
                    analytical_mask.append(i)

                index_map.append((float(k), float(h)))
                i += 1

        return (
            jnp.array(states),
            jnp.array(labels),
            jnp.array(analytical_mask).astype(int),
            index_map,
        )

    def __repr__(self):
        repr_str = f"Input Class ({self.__class__.__name__}):"
        repr_str += f"\n  N        : {self.num_qubits}"
        repr_str += f"\n  side     : {self.side}"

        return repr_str
