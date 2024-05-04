import os
import torch
import numpy as np
from config import Config
from models import get_model
from sources import get_source_velocity


# create experiment config containing all hyperparameters
cfg = Config('test')

# create network and training agent
fluid = get_model(cfg)

# iterate
Ek_list = []
for t in range(cfg.n_timesteps):
    fluid.load_ckpt(t + 1)
    with torch.no_grad():
        grid_vel, grid_samples = fluid.sample_velocity_field(cfg.vis_resolution, to_numpy=False, return_samples=True)
    grid_vel = grid_vel.detach().cpu().numpy()
    Ek = 0.5 * np.sum(grid_vel ** 2)
    Ek_list.append(Ek)

source_func = get_source_velocity(cfg.src, cfg.src_start_frame)
grid_vel_src = source_func(grid_samples).detach().cpu().numpy()
Ek_src = 0.5 * np.sum(grid_vel_src ** 2)

print('Ek src:')
print(Ek_src)
print('Ek list:')
print(Ek_list)

save_path = os.path.join(cfg.results_dir, f'Ek_r{cfg.vis_resolution}.txt')
with open(save_path, 'w') as fp:
    print(f'Ek src:\n{Ek_src}', file=fp)
    print('Ek list:', file=fp)
    for Ek in Ek_list:
        print(Ek, file=fp)
