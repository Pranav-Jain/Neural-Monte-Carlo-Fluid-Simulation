import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib
# import cmasher as cmr

def draw_scalar_field2D(arr, vmin=None, vmax=None, to_array=False, figsize=(3, 3), cmap='Greys', colorbar=False, clip=None):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(2000*px, 2000*px))
    orig_map=plt.cm.get_cmap(cmap)  
    reversed_map = orig_map.reversed() 
    # matplotlib.rcParams['axes.edgecolor'] = 'ffffff'
    cax1 = ax.pcolormesh(arr, shading='gouraud', vmin=vmin, vmax=vmax, cmap=cmap)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    # for axis in ['top','bottom','left','right']:
    #     ax.spines[axis].set_linewidth(6)
    # circle = plt.Circle([-1, 0], 4./30., fill = False, linewidth=4)
    # ax.add_artist(circle)
    # fig.tight_layout()
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-1, 1)
    ax.set_axis_off()
    ax.set_aspect('equal')
    plt.axis('equal')
    return fig
    return figure2array(fig)

def draw_vector_field2D(u, v, x=None, y=None, c=None, r=None, tag=None, to_array=False, figsize=(5, 5), p=None):
    assert u.shape == v.shape
    # s = 5 * (u.shape[0] // 50 + 1)
    # fig, ax = plt.subplots(figsize=(s, s))
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(2000*px, 2000*px))
    if x is None:
        # buggy
        raise NotImplementedError
        indices = np.indices(u.shape)
        # ax.quiver(indices[1], indices[0], u, v, scale=u.shape[0], scale_units='width')
        ax.quiver(indices[0], indices[1], u, v, scale=u.shape[0], scale_units='width')
    else:
        # ax.quiver(y, x, u, v, scale=u.shape[0], scale_units='width')
        ax.quiver(x, y, u, v, scale=u.shape[0], scale_units='width')
        if c is not None and r is not None:
            circle = plt.Circle(c, r, fill = False)
            ax.add_artist(circle)
        if p is not None:
            cmap = ax.pcolormesh(x, y, p, shading='auto', cmap=plt.cm.jet, alpha=0.2)
            fig.colorbar(cmap)
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
    ax.set_axis_off()
    ax.set_aspect('equal')
    plt.axis('equal')
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
