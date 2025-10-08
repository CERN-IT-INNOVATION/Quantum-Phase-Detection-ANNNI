from pathlib import Path

import matplotlib.pyplot as plt

from ..annni.plot import set_layout


def loss_curve(loss_history, path=""):
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Compression Score")
    plt.title("Loss Curve")
    if path:
        plt.savefig(Path(path) / "losscurve.pdf")
    plt.show()


def compress(compression_img, num_qubits, khs_train=[], path=""):
    side = compression_img.shape[0]
    # Plot the classification results
    plt.figure(figsize=(4.5, 4))

    plt.imshow(
        compression_img,
        aspect="auto",
        origin="lower",
        extent=[0, 1, 0, 2],
        vmin=0,
    )
    plt.colorbar()
    for i, kh in enumerate(khs_train):
        label = "Training Point" if i == 0 else None
        plt.scatter(
            [kh[0] + 0.3 / side],
            [kh[1] + 0.5 / side],
            color="red",
            marker="x",
            label=label,
        )
    set_layout(f"Compression: L={num_qubits}")

    if path:
        plt.savefig(Path(path) / "compression.pdf")
    plt.show()
