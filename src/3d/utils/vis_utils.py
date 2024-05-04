import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
# import cmasher as cmr

def draw_scalar_field2D(points, arr, savepath, vmin=None, vmax=None, to_array=False, figsize=(3, 3), cmap='bwr', colorbar=True):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    sc = ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=arr, cmap=cmap, alpha=0.02, vmin=None, vmax=None)
    fig.tight_layout()
    if colorbar:
        plt.colorbar(sc)
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close("all")
    
def draw_vector_field2D(points, vel, savepath, x=None, y=None, c=None, r=None, tag=None, to_array=False, figsize=(5, 5), p=None):
    density = np.linalg.norm(vel, axis=1)
    density[density<1e-6] = None
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=density, cmap='Greys', alpha=0.02, vmin=0, vmax=None)
    fig.tight_layout()
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close("all")

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # sc = ax.quiver(points[..., 0], points[..., 1], points[..., 2], vel[..., 0], vel[..., 1], vel[..., 2], length=0.1)
    # fig.tight_layout()
    # plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=100)
    # plt.close("all")


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


def save_figure_nopadding(fig, save_path, close=True):
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
    # plt.savefig(save_path, bbox_inches='tight')
    if close:
        plt.close("all")


def save_figure(fig, save_path, close=True):
    # plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)
    plt.savefig(save_path, bbox_inches='tight')
    if close:
        plt.close("all")


def frames2gif(src_dir, save_path, fps=24):
    filenames = sorted([x for x in os.listdir(src_dir) if x.endswith('.png')])
    img_list = [imageio.imread(os.path.join(src_dir, name)) for name in filenames]
    imageio.mimsave(save_path, img_list, fps=fps, loop=0)
