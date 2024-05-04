import os
from tqdm import tqdm
from config import Config

cfg = Config("recap")

if cfg.pde == "advection":
    from advection import Advection1DModel as neuralModel
elif cfg.pde == "fluid":
    from fluid import Fluid2DModel as neuralModel
else:
    raise NotImplementedError
model = neuralModel(cfg)

output_folder = os.path.join(cfg.exp_dir, cfg.output)
os.makedirs(output_folder, exist_ok=True)

for t in tqdm(range(cfg.n_timesteps + 1)):
    try:
        model.load_ckpt(t)
    except Exception as e:
        print(f"Load checkpoint t={t} failed.\n {e}")
        break

    model.write_output(output_folder)
