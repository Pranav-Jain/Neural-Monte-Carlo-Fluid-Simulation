from PIL import Image
from scipy.special import erf
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def draw_vector_field2D(vel, coords):
    u, v = vel[..., 0], vel[..., 1]
    x, y = coords[..., 0], coords[..., 1]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.quiver(x, y, u, v, scale=u.shape[0], scale_units='width')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    return fig


def draw_scalar_field2D(arr, vmin=None, vmax=None, cmap=None):
    multi = max(arr.shape[0] // 512, 1)
    fig, ax = plt.subplots(figsize=(5 * multi, 5 * multi))
    cax1 = ax.matshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(cax1, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def draw_curl(curl):
    """draw 2D curl(vorticity) field"""
    curl = (erf(curl) + 1) / 2 # map range to 0~1
    img = cm.bwr(curl)
    img = (img * 255).astype('uint8')
    return img


def draw_magnitude(mag):
    """draw 2D velocity magnitude"""
    mag = erf(mag)
    img = cm.Blues(mag)
    img = (img * 255).astype('uint8')
    return img


def save_numpy_img(arr, save_path):
    img = Image.fromarray(arr)
    if save_path is not None:
        img.save(save_path)
    return img


def save_figure(fig, save_path, close=True):
    plt.savefig(save_path, bbox_inches='tight')
    if close:
        plt.close(fig)
