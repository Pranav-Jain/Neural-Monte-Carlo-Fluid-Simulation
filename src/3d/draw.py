import os
import torch
from tqdm import tqdm
from config import Config
from models import get_model
from utils.vis_utils import save_figure, frames2gif
from sources import circle_obstable_functions

attr = "vorticity"

# create experiment config containing all hyperparameters
cfg = Config('test')

# create network and training agent
fluid = get_model(cfg)

save_dir = os.path.join(cfg.results_dir, f'{attr}_{cfg.vis_resolution}')
os.makedirs(save_dir, exist_ok=True)

center = (-cfg.scene_size[0] / 4, 0)
radius = cfg.scene_size[0] / 30
sign_func, bound_func = circle_obstable_functions(center, radius)
fluid.add_obstacle(sign_func, bound_func)

# iterate
for t in tqdm(range(200 + 1)):
    if t == 0:
        fluid.load_ckpt("add_source")
    else:
        fluid.load_ckpt(t)
    # with torch.no_grad():
    fig = fluid.draw(attr, cfg.vis_resolution, vmin=-5, vmax=5)
    save_path = os.path.join(save_dir, f'{attr}_t{t:03d}.png')
    save_figure(fig, save_path)

save_path = os.path.join(save_dir, f'{attr}_anim.gif')
frames2gif(save_dir, save_path, fps=cfg.fps)
