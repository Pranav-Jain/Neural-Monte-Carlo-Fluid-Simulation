import os
import torch
from functools import partial
import numpy as np
import sys
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

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

interpolate = partial(map_coordinates, 
            order=1, prefilter=False, mode='constant', cval=0)

N = 50
dt = 0.001

d_grid = np.zeros((N, N))
grid_coords = np.indices(d_grid.shape)
grid_coords = grid_coords.transpose((1, 2, 0)).astype(float)
grid_coords_domain = (grid_coords)/N * 2*np.pi

d_grid = tlgn_density(torch.from_numpy(grid_coords/N))

true = np.array([np.cos(grid_coords_domain[...,0])*np.sin(grid_coords_domain[..., 1]), -np.sin(grid_coords_domain[...,0])*np.cos(grid_coords_domain[..., 1])])
true = true.transpose((1, 2, 0))
print(true.shape)

filename = sys.argv[1]

vel = np.load(filename)

print(vel.shape)

error = []
t=0
for v in vel:
    t+=1
    e = np.linalg.norm(v - true, axis=2)**2
    e = np.mean(e)
    error.append(e)
    print(e)

    i_back = grid_coords_domain[..., 0] - dt * v[..., 0]
    j_back = grid_coords_domain[..., 1] - dt * v[..., 1]
    back_pos = (np.stack([i_back, j_back]))/(2*np.pi) * N
    d_grid = interpolate(d_grid, back_pos)

    fig = draw_scalar_field2D(np.linalg.norm(v - true, axis=2), to_array=False, vmin=None, vmax=None, cmap='viridis', colorbar=True)
    save_path = os.path.join(save_dir, f'error_t{t:03d}.png')
    save_figure(fig, save_path)
    plt.close("all")

print("Error: ", np.sqrt(np.mean(error)))
np.savetxt("error_pideeponet", error)