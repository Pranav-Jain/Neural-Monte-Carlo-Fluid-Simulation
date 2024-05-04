import os
from tqdm import tqdm
import torch
import numpy as np
from config import Config
from models import get_model
from functools import partial
from utils.vis_utils import draw_scalar_field2D, save_figure, frames2gif, draw_vector_field2D
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

# Read obj with lines
def read_obj(filename, d2=True):
    v = []
    l = []
    file = open(filename, 'r')
    lines = file.readlines()

    for line in lines:
        line = line.split(" ")
        if line[0] == 'v':
            if d2==True:
                v.append([float(line[1]), float(line[2])])
            else:
                v.append([float(line[1]), float(line[2]), float(line[3])])
        elif line[0] == 'l':
            l.append([int(line[1])-1, int(line[2])-1])
    
    file.close()
    return np.array(v), np.array(l)

def get_scene_size(filename):
    v, l = read_obj(filename)
    min_x = np.min(v[:, 0])
    max_x = np.max(v[:, 0])
    min_y = np.min(v[:, 1])
    max_y = np.max(v[:, 1])

    # scene_size = [max_x - min_x, max_y - min_y]
    scene_size = [min_x, max_x, min_y, max_y]
    
    return scene_size

def tlgn_density(samples: torch.FloatTensor, scene_size):
    A = 1
    a = 1
    B = -1
    b = 1
    x = ((samples[..., 0]-scene_size[0])/(scene_size[1] - scene_size[0])) * 2*np.pi
    y = ((samples[..., 1]-scene_size[2])/(scene_size[3] - scene_size[2])) * 2*np.pi
    u = A * torch.sin(a * x) * torch.cos(b * y)
    v = B * torch.cos(a * x) * torch.sin(b * y)
    vel = torch.stack([u, v], dim=-1)

    out = np.linalg.norm(vel, axis=2)
    print(out.shape)
    # print(out)
    return out

def jpipe_density(samples: torch.FloatTensor, scene_size):
    vel = torch.zeros_like(samples)
    mask = samples[..., 0] < 1.9
    vel[..., 0][mask] = cfg.karman_vel

    d = torch.sqrt((samples[..., 0]-2)**2 + (samples[..., 1]-2)**2)
    mask1 = (samples[..., 0]>=0.0) & (samples[..., 0]<=2.0) & (samples[..., 1]>=0.0) & (samples[..., 1]<=1.0)
    mask2 = (samples[..., 0]>=3.0) & (samples[..., 0]<=4.0) & (samples[..., 1]>=2.0) & (samples[..., 1]<=4.0)
    mask3 = (d >= 1.0) & (d <= 2.0) & (samples[..., 0]>=2.0) & (samples[..., 1]<=2.0)
    mask = mask1 | mask2 | mask3
    vel[~mask] = 0.0

    out = np.linalg.norm(vel, axis=2)
    print(out.shape)
    # print(out)
    return out

# create experiment config containing all hyperparameters
cfg = Config()
# cfg.scene_size = get_scene_size('./jpipe.obj')
cfg.scene_size = get_scene_size('./square.obj')

print(cfg.scene_size)

# create network and training agent
fluid = get_model(cfg)

save_dir = os.path.join(cfg.results_dir, f'density_{cfg.vis_resolution}')
os.makedirs(save_dir, exist_ok=True)

N = 1000
dt = cfg.dt
interpolate = partial(map_coordinates, 
            order=1, prefilter=False, mode='constant', cval=0)

d_grid = np.zeros((N, N))
grid_coords = np.indices(d_grid.shape)
grid_coords = grid_coords.transpose((1, 2, 0)).astype(float)
grid_coords_domain = (grid_coords)/N * 2*np.pi

true = np.array([np.sin(grid_coords_domain[...,0])*np.cos(grid_coords_domain[..., 1]), -np.cos(grid_coords_domain[...,0])*np.sin(grid_coords_domain[..., 1])])
true = true.transpose((1, 2, 0))
print(true.shape)
grid_coords = (grid_coords)/N*(cfg.scene_size[1] - cfg.scene_size[0]) + cfg.scene_size[0]

# grid_coords1 = (grid_coords + 0.5) / N * 2 - 1.0 # normalize to (-1, 1)


d_grid = tlgn_density(torch.from_numpy(grid_coords), cfg.scene_size)
# d_grid = jpipe_density(torch.from_numpy(grid_coords), cfg.scene_size)

# iterate
grid_coords_torch = torch.tensor((grid_coords), dtype=torch.float32).cuda()
print(grid_coords_torch.shape)
error = []
for t in tqdm(range(cfg.max_n_iters)):
    fluid.load_ckpt(t)
    with torch.no_grad():
        grid_vel = fluid.velocity_field(grid_coords_torch).detach().cpu().numpy()

    fig = draw_scalar_field2D(d_grid, to_array=False, vmin=None, vmax=None, cmap='Blues', colorbar=True)
    # fig = draw_vector_field2D(grid_vel[...,0],grid_vel[...,1],grid_coords[...,0],grid_coords[...,1], to_array=False)
    save_path = os.path.join(save_dir, f'error_t{t:03d}.png')
    # fig = draw_vector_field2D(true[...,0],true[...,1],grid_coords[...,0],grid_coords[...,1], to_array=False)
    # save_path = os.path.join(save_dir
    save_figure(fig, save_path)
    plt.close("all")

    e = np.mean(np.linalg.norm(grid_vel - true, axis=2)**2)
    print(e)
    error.append(e)

print(np.mean(error))
np.savetxt("error_ours.txt", error)
# save_path = os.path.join(save_dir, 'density_anim.gif')
# frames2gif(save_dir, save_path, fps=cfg.fps)
