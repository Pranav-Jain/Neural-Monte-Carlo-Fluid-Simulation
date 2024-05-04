import os
import numpy as np
from config import Config
from model import NeuralFluid
from sources import get_source_velocity, get_source_density
from utils.vis_utils import save_figure, frames2gif
from utils.file_utils import ensure_dirs


# create experiment config containing all hyperparameters
cfg = Config()

# create network and training agent
fluid = NeuralFluid(cfg)

fluid.load_ckpt(path=cfg.ckpt)

# add source
source_func = get_source_velocity(cfg.src)

# train
fluid.train(source_func)
