import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
# import cmasher as cmr


def draw_scalar_field2D(arr, vmin=None, vmax=None, to_array=False, cmap=None):
    multi = max(arr.shape[0] // 512, 1)
    fig, ax = plt.subplots(figsize=(5 * multi, 5 * multi))
    if cmap is None:
        cax1 = ax.matshow(arr, vmin=vmin, vmax=vmax)
    else:
        cax1 = ax.matshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
    # fig.colorbar(cax1, ax=ax, fraction=0.046, pad=0.04)
    ax.set_axis_off()
    fig.tight_layout()
    if not to_array:
        return fig
    return figure2array(fig)


def draw_vector_field2D(u, v, x=None, y=None, tag=None, to_array=False):
    assert u.shape == v.shape
    s = 5 * (u.shape[0] // 50 + 1)
    # fig, ax = plt.subplots(figsize=(s, s))
    fig, ax = plt.subplots(figsize=(5, 5))
    if x is None:
        # buggy
        raise NotImplementedError
        indices = np.indices(u.shape)
        # ax.quiver(indices[1], indices[0], u, v, scale=u.shape[0], scale_units='width')
        ax.quiver(indices[0], indices[1], u, v, scale=u.shape[0], scale_units='width')
    else:
        # ax.quiver(y, x, u, v, scale=u.shape[0], scale_units='width')
        ax.quiver(x, y, u, v, scale=u.shape[0], scale_units='width')
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
    if tag is not None:
        ax.text(-1, -1, tag, fontsize=12)
    fig.tight_layout()
    if not to_array:
        return fig
    return figure2array(fig)


# def draw_vorticity_field2D(curl, x, y, to_array=False):
#     fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
#     ax.contourf(
#         x,
#         y,
#         curl,
#         cmap=cmr.arctic,
#         levels=100,
#     )
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     fig.tight_layout()
#     if not to_array:
#         return fig
#     return figure2array(fig)


def figure2array(fig):
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def save_figure(fig, save_path, close=True, axis_off=False):
    if axis_off:
        plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    if close:
        plt.close(fig)


def frames2gif(src_dir, save_path, fps=24):
    filenames = sorted([x for x in os.listdir(src_dir) if x.endswith('.png')])
    img_list = [imageio.imread(os.path.join(src_dir, name)) for name in filenames]
    imageio.mimsave(save_path, img_list, fps=fps)
