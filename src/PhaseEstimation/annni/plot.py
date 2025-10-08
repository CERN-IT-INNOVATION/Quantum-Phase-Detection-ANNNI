import matplotlib.pyplot as plt
import pennylane.numpy as np
from matplotlib.colors import ListedColormap

from .lines import b1, get_phase, paraanti, paraferro, peshel_emery

PHASES3 = [
    "Ferromagnetic",
    "Paramagnetic",
    "Antiphase",
    "Trash Class",
]
PHASES4 = ["Ferromagnetic", "Paramagnetic", "Antiphase", "Floatingphase"]
COLORS = [
    "#80bfff",
    "#80f090",
    "#fff2a8",
    "#da8080",
]


def set_layout(title="Phase-Space"):
    # X values for different phase boundaries
    xl = np.linspace(0, 0.5, 50)
    xpe = np.linspace(0.12, 0.5, 50)
    xr = np.linspace(0.5, 1, 50)

    # Plot the transition lines
    plt.plot(xpe, peshel_emery(xpe, y_cap=2), ls="--", color="black")
    plt.plot(xr, paraanti(xr), color="black")
    plt.plot(xl, paraferro(xl), color="black")
    plt.plot(xr, b1(xr), color="black", ls="--")
    plt.plot([], [], color="black", label="Transition Lines")

    plt.legend()
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("h")

    # Explicitly set the correct y-limits
    plt.ylim(0, 2)


def show_phases(side: int = 40):
    p_k = np.linspace(0, 1, side)
    p_h = np.linspace(0, 2, side)

    img = np.zeros((side, side))
    for x, k in enumerate(p_k):
        for y, h in enumerate(p_h):
            phase = get_phase(k=k, h=h)
            if phase == 2 and (h > b1(np.array([k]))):
                phase = 3
            img[x, y] = phase

    # Create a custom colormap
    plt.figure(figsize=(4, 4))

    for color, phase in zip(COLORS, PHASES4):
        plt.scatter([], [], color=color, label=phase)
    cmap = ListedColormap(COLORS)
    plt.imshow(
        np.flip(np.rot90(img, k=-1), axis=1),
        cmap=cmap,
        aspect="auto",
        origin="lower",
        extent=(0, 1, 0, 2),
    )
    set_layout()
