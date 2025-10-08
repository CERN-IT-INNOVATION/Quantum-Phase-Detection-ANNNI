from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from pennylane import numpy as np

from ..annni.plot import COLORS, PHASES3, set_layout


def loss_curve(loss_history, T=None, path=""):
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    if T is None:
        plt.ylabel("Cross-Entropy Loss")
    else:
        plt.ylabel(f"Cross-Entropy Loss (T = {T})")
    plt.title("Loss Curve")
    if path:
        plt.savefig(Path(path) / "losscurve.pdf")
    plt.show()


def predict_all(predictions, num_qubits, T, path=""):
    # Ensure reshaping is valid
    img = np.argmax(predictions, axis=2)
    cmap = ListedColormap(COLORS)

    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(4, 4), constrained_layout=True)
    plt.imshow(
        img,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        origin="lower",
        extent=[0, 1, 0, 2],
    )

    for phase in np.unique(img):
        plt.scatter(
            [],
            [],
            color=COLORS[phase],
            label=PHASES3[phase],
            edgecolors="black",
        )

    plt.legend()
    set_layout(f"Classification: L={num_qubits} | T={T}")
    if path:
        plt.savefig(Path(path) / "pred.pdf")
    plt.show()


def predict_single(predictions, num_qubits, T, path=""):
    for i, label in enumerate(
        ["Paramagnetic", "Ferromagnetic", "Antiphase", "Trash Class"]
    ):
        img = predictions[:, :, i]
        plt.figure(figsize=(4.5, 4))
        vargs = {"vmin": 0, "vmax": 1}
        plt.imshow(
            img,
            cmap="viridis",
            aspect="auto",
            origin="lower",
            extent=[0, 1, 0, 2],
            **vargs,
        )
        set_layout(f"{label}: L={num_qubits} | T={T}")
        plt.colorbar()
        if path:
            plt.savefig(Path(path) / f"pred_{label.replace(' ', '')}.pdf")
        plt.show()
