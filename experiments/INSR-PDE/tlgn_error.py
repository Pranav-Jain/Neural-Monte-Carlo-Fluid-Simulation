import os
import torch
import numpy as np
import sys
import glob
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from functools import partial

def tlgn_density(samples: torch.FloatTensor):
    A = 1
    a = 1
    B = -1
    b = 1
    x = samples[..., 0] * 2*np.pi
    y = samples[..., 1] * 2*np.pi
    u = A * torch.sin(a * x) * torch.cos(b * y)
    v = B * torch.cos(a * x) * torch.sin(b * y)
    vel = torch.stack([u, v], dim=-1)

    out = np.linalg.norm(vel, axis=2)
    print(out.shape)
    return out

def save_figure(fig, save_path, close=True):
    plt.savefig(save_path, bbox_inches='tight')
    if close:
        plt.close("all")

save_dir = os.path.join(os.curdir, f'density')
os.makedirs(save_dir, exist_ok=True)

def draw_scalar_field2D(arr, vmin=None, vmax=None, to_array=False, figsize=(3, 3), cmap='Greys', colorbar=False):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(2000*px, 2000*px))
    orig_map=plt.cm.get_cmap(cmap)  
    reversed_map = orig_map.reversed()
    cax1 = ax.pcolormesh(arr, shading='gouraud', vmin=vmin, vmax=vmax, cmap='OrRd', clim=(0, 0.2))
    ax.set_axis_off()
    ax.set_aspect('equal')
    plt.axis('equal')
    if colorbar:
        plt.colorbar(cax1)
    if not to_array:
        return fig
    
def draw_vector_field2D(u, v, x=None, y=None, c=None, r=None, tag=None, to_array=False, figsize=(5, 5), p=None):
    assert u.shape == v.shape
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    fig, ax = plt.subplots(figsize=(2000*px, 2000*px))
    if x is None:
        # buggy
        raise NotImplementedError
    else:
        ax.quiver(x, y, u, v, scale=u.shape[0], scale_units='width')
        if c is not None and r is not None:
            circle = plt.Circle(c, r, fill = False)
            ax.add_artist(circle)
        if p is not None:
            cmap = ax.pcolormesh(x, y, p, shading='auto', cmap='plt.cm.jet', alpha=0.2)
            fig.colorbar(cmap)
    ax.set_axis_off()
    if tag is not None:
        ax.text(-1, -1, tag, fontsize=12)
    fig.tight_layout()
    if not to_array:
        return fig

interpolate = partial(map_coordinates, 
            order=1, prefilter=False, mode='constant', cval=0)

N = 100
dt = 0.001

d_grid = np.zeros((N, N))
grid_coords = np.indices(d_grid.shape)
grid_coords = grid_coords.transpose((1, 2, 0)).astype(float)
grid_coords_domain = (grid_coords)/N * 2*np.pi

d_grid = tlgn_density(torch.from_numpy(grid_coords/N))

true = np.array([np.sin(grid_coords_domain[...,0])*np.cos(grid_coords_domain[..., 1]), -np.cos(grid_coords_domain[...,0])*np.sin(grid_coords_domain[..., 1])])
true = true.transpose((1, 2, 0))
print(true.shape)

dir = sys.argv[1]
files = glob.glob(dir + "/*.npy")
files = np.sort(files)
error = []
t = 0
for i in files:
    t+=1
    v = np.load(i)
    e = np.mean(np.linalg.norm(v - true, axis=2))**2
    error.append(e)

    i_back = grid_coords_domain[..., 0] - dt * v[..., 0]
    j_back = grid_coords_domain[..., 1] - dt * v[..., 1]
    back_pos = (np.stack([i_back, j_back]))/(2*np.pi) * N
    d_grid = interpolate(d_grid, back_pos)

    fig = draw_scalar_field2D(np.linalg.norm(v - true, axis=2), to_array=False, vmin=None, vmax=None, cmap='viridis', colorbar=True)
    save_path = os.path.join(save_dir, f'error_t{t:03d}.png')
    save_figure(fig, save_path)

np.savetxt("error_INSR.txt", error)
print("Error: ", np.sqrt(np.mean(error)))
