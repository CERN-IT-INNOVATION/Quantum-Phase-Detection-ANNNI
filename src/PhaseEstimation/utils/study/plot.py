import numpy as np
from matplotlib import pyplot as plt


def imgs(imgs, titles, tag, _range=None, cmap="viridis"):
    # Ensure imgs and titles are lists (even if single elements)
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    if not isinstance(titles, (list, tuple)):
        titles = [titles]

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    # If there's only one subplot, wrap it in a list for uniform handling
    if n == 1:
        axes = [axes]

    # Compute global vmin/vmax
    if _range is None:
        vmin = np.min([np.min(img) for img in imgs])
        vmax = np.max([np.max(img) for img in imgs])
    else:
        vmin, vmax = _range

    # Plot each image
    for ax, img, title in zip(axes, imgs, titles):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")

    # Shared colorbar
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025, pad=0.04)

    # Title and save
    fig.suptitle(tag, fontsize=16)
