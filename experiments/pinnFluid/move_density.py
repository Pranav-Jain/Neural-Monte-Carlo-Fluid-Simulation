import os
from tqdm import tqdm
import torch
import numpy as np
from config import Config
from model import NeuralFluid
from sources import get_source_density
from functools import partial
from utils.vis_utils import draw_scalar_field2D, save_figure, frames2gif
from scipy.ndimage import map_coordinates

# create experiment config containing all hyperparameters
cfg = Config('test')
print(cfg.src)

# create network and training agent
fluid = NeuralFluid(cfg)
fluid.load_ckpt("final")

save_dir = os.path.join(cfg.results_dir, f'density_{cfg.vis_resolution}')
os.makedirs(save_dir, exist_ok=True)

N = 1000
dt = 0.001
interpolate = partial(map_coordinates, 
            order=1, prefilter=False, mode='constant', cval=0)

d_grid = np.zeros((N, N))
grid_coords = np.indices(d_grid.shape)
grid_coords = grid_coords.transpose((1, 2, 0)).astype(float)
grid_coords = (grid_coords + 0.5) / N * 2 - 1.0 # normalize to (-1, 1)

source_func = get_source_density(cfg.src)
d_grid = source_func(torch.from_numpy(grid_coords)).numpy()

# iterate
grid_coords_torch = torch.tensor(grid_coords, dtype=torch.float32).cuda()
time = torch.zeros(grid_coords_torch.shape[:-1]).unsqueeze(-1).cuda()
all_d_grids = []
for t in tqdm(range(100 + 1)):
    if t > 0:
        time += dt
        with torch.no_grad():
            grid_vel = fluid.velocity_field(grid_coords_torch, time).detach().cpu().numpy()

        i_back = grid_coords[..., 0] - dt * grid_vel[..., 0]
        j_back = grid_coords[..., 1] - dt * grid_vel[..., 1]

        back_pos = (np.stack([i_back, j_back]) + 1.0) / 2 * N - 0.5
        d_grid = interpolate(d_grid, back_pos)

    fig = draw_scalar_field2D(d_grid, to_array=False, vmin=0, vmax=1, cmap="Blues")
    save_path = os.path.join(save_dir, f'density_t{t:03d}.png')
    save_figure(fig, save_path)

    all_d_grids.append(d_grid)

save_path = os.path.join(save_dir, 'density_anim.gif')
frames2gif(save_dir, save_path, fps=cfg.fps)

save_path = os.path.join(save_dir, 'density_grids.npy')
all_d_grids = np.stack(all_d_grids, axis=0)
np.save(save_path, all_d_grids)
