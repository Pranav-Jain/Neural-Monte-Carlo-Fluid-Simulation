import os
from tqdm import tqdm
import torch
import numpy as np
from config import Config
from model import NeuralFluid
from sources import get_source_velocity
from functools import partial
from utils.vis_utils import draw_scalar_field2D, save_figure, frames2gif
from scipy.ndimage import map_coordinates

# create experiment config containing all hyperparameters
cfg = Config('test')

# create network and training agent
fluid = NeuralFluid(cfg)
fluid.load_ckpt("final")

save_dir = os.path.join(cfg.results_dir, f'velocity_{cfg.vis_resolution}')
os.makedirs(save_dir, exist_ok=True)

N = 50
dt = 0.001

# d_grid = np.zeros((N, N))
# grid_coords = np.indices(d_grid.shape)
# grid_coords = grid_coords.transpose((1, 2, 0)).astype(float)
# grid_coords = (grid_coords + 0.5) / N * 2 - 1.0 # normalize to (-1, 1)

source_func = get_source_velocity(cfg.src)
fluid.set_source(source_func)
# d_grid = source_func(torch.from_numpy(grid_coords)).numpy()


# iterate
# grid_coords_torch = torch.tensor(grid_coords, dtype=torch.float32).cuda()
# time = torch.zeros(grid_coords_torch.shape[:-1]).unsqueeze(-1).cuda()


# init_vel = fluid.velocity_field(grid_coords_torch, time).detach().cpu().numpy()
# print(init_vel.max())

all_d_grids = []
grid_vel_list = []
time = 0
for t in tqdm(range(cfg.n_timesteps + 1)):
    with torch.no_grad():
        grid_vel = fluid.sample_velocity_field(N, time, to_numpy=True)
        # grid_vel = fluid.velocity_field(grid_coords_torch, time).detach().cpu().numpy()
        print(grid_vel.max())
        print(grid_vel.shape)

    grid_vel_list.append(grid_vel)
    time += dt

grid_vel_list = np.stack(grid_vel_list, axis=0)

save_path = os.path.join(save_dir, 'velocity_grids.npy')
np.save(save_path, grid_vel_list)
