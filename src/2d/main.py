import os
import numpy as np
from functools import partial
from config import Config
from models import get_model
from sources import get_source_velocity, circle_obstable_functions, jpipe_obstable_functions
from utils.vis_utils import save_figure, frames2gif
from utils.file_utils import ensure_dirs
import matplotlib.pyplot as plt
import json
import gpytoolbox
import torch

torch.cuda.empty_cache()

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
    obstacle_lines = []
    for i in range(l.shape[0]):
        if np.all(v[l[i, 0]] == v[l[i, 1]]):
            continue
        if (v[l[i,0], 0] > min_x and v[l[i,0], 0] < max_x and v[l[i,0], 1] > min_y and v[l[i,0], 1] < max_y) or (v[l[i,1], 0] > min_x and v[l[i,1], 0] < max_x and v[l[i,1], 1] > min_y and v[l[i,1], 1] < max_y):
            obstacle_lines.append(l[i])
    obstacle_lines = np.array(obstacle_lines)

    if len(obstacle_lines) > 0:
        obstacle_vertices, obstacle_lines = gpytoolbox.remove_unreferenced(v, obstacle_lines)
        obstacle_vertices, _, _, obstacle_lines = gpytoolbox.remove_duplicate_vertices(obstacle_vertices, faces=obstacle_lines)

        return scene_size, v, obstacle_vertices[:, :2], obstacle_lines
    
    return scene_size, v, [], []


# create experiment config containing all hyperparameters
cfg = Config()

# vis results save folder
vis_vel_dir = os.path.join(cfg.results_dir, 'velocity')
vis_vor_dir = os.path.join(cfg.results_dir, 'vorticity')
txt_dir = os.path.join(cfg.results_dir, 'txt')
ensure_dirs([vis_vel_dir, vis_vor_dir, txt_dir])

vis_mag_dir = os.path.join(cfg.results_dir, 'magnitude')
vis_pressure_dir = os.path.join(cfg.results_dir, 'pressure')
ensure_dirs([vis_mag_dir, vis_pressure_dir])

try:
    f = open(cfg.wost_json)
    wost_data = json.load(f)
    f.close()
except:
    print("wost.json Not found")
    exit()
cfg.scene_size, vertices, obstacle_vertices, obstacle_lines = get_scene_size(wost_data["scene"]["boundary"])

print("bbox: ", cfg.scene_size)

# create network and training agent
fluid = get_model(cfg)

if cfg.src == 'karman':
    if cfg.obstacle == "one_cylinder":
        mask = (vertices[..., 0] == cfg.scene_size[2])
        fluid.vertices = torch.Tensor(vertices[~mask]).to(fluid.device)

        fluid.obstacle_vertices = obstacle_vertices
        obs_c = np.mean(obstacle_vertices, axis=0)
        obs_r = np.mean(np.linalg.norm(obstacle_vertices - obs_c, axis=1))+wost_data["output"]["boundaryDistanceMask"]
        
        print("obs_c: ", obs_c)
        print("obs_radius: ", obs_r)
        
        center = np.array([obs_c[0], obs_c[1]])
        radius = obs_r
        sign_func = circle_obstable_functions(center, radius)
        fluid.add_obstacle(sign_func)
        fluid.center = center
        fluid.radius = radius

if cfg.src == 'jpipe':
    sign_func = jpipe_obstable_functions()
    fluid.add_obstacle(sign_func)

# load checkpoints
if cfg.ckpt > 0:
    fluid.load_ckpt(cfg.ckpt)
else:
    source_func = get_source_velocity(cfg.src, cfg.src_start_frame)
    if cfg.src == 'karman':
        source_func = partial(source_func, karman_vel=cfg.karman_vel, obs_func=sign_func, scene_size=cfg.scene_size, eps=cfg.bdry_eps)
    if cfg.src == 'taylorgreen':
        source_func = partial(source_func, scene_size=cfg.scene_size)
    if cfg.src == 'jpipe':
        source_func = partial(source_func, karman_vel=cfg.karman_vel, obs_func=sign_func, eps=cfg.bdry_eps)
    fluid.add_source('velocity', source_func, is_init=True)
    
    save_path_txt_v = os.path.join(txt_dir, f'velocity_values_t{fluid.timestep:03d}.txt')
    save_path_txt_s = os.path.join(txt_dir, f'velocity_samples_t{fluid.timestep:03d}.txt')
    fig = fluid.draw_velocity(cfg.vel_vis_resolution, save_path_txt_v, save_path_txt_s)
    save_path_png = os.path.join(vis_vel_dir, f'velocity_t{fluid.timestep:03d}.png')
    save_figure(fig, save_path_png)
    
    save_path_txt_v = os.path.join(txt_dir, f'vorticity_values_t{fluid.timestep:03d}.txt')
    save_path_txt_s = os.path.join(txt_dir, f'vorticity_samples_t{fluid.timestep:03d}.txt')
    fig = fluid.draw_vorticity(cfg.vis_resolution, save_path_txt_v, save_path_txt_s)
    save_path = os.path.join(vis_vor_dir, f'vorticity_t{fluid.timestep:03d}.png')
    save_figure(fig, save_path)
    
    # try:
    #     fluid.load_ckpt('add_source')
    #     print("load pretrained model that fits initial condition.")
    # except Exception as e:
    #     # get source function
    #     if cfg.use_density:
    #         source_func = get_source_density(cfg.src)
    #         fluid.add_source_density('density', source_func)
    #     source_func = get_source_velocity(cfg.src, cfg.src_start_frame)
    #     if cfg.src == 'karman':
    #         source_func = partial(source_func, karman_vel=cfg.karman_vel, obs_func=sign_func)
    #     fluid.add_source('velocity', source_func, is_init=True)
        
    #     fig = fluid.draw('velocity', cfg.vis_resolution)
    #     save_path = os.path.join(vis_vel_dir, f'velocity_t{fluid.timestep:03d}.png')
    #     save_figure(fig, save_path)
    #     fig = fluid.draw('vorticity', cfg.vis_resolution)
    #     save_path = os.path.join(vis_vor_dir, f'vorticity_t{fluid.timestep:03d}.png')
    #     save_figure(fig, save_path)

# start simulation
energy = []
timestep = []
# fluid.reset_weights()
if cfg.src == 'karman':
    cfg.bdry_eps /= 2
    fluid.bdry_eps /= 2
for t in range(cfg.n_timesteps):
    # if t == 1:
    #     fluid.max_n_iters = 3000
    #     cfg.max_n_iters = 3000

    fluid.timestep += 1
    if t > 0 and t < cfg.src_duration:
        fluid.add_source('velocity', source_func, is_init=False)

    # time-stepping
    print("timestep:", fluid.timestep)
    fluid.step()

    # save visualization
    save_path_txt_v = os.path.join(txt_dir, f'velocity_values_t{fluid.timestep:03d}.txt')
    save_path_txt_s = os.path.join(txt_dir, f'velocity_samples_t{fluid.timestep:03d}.txt')
    fig = fluid.draw_velocity(cfg.vel_vis_resolution, save_path_txt_v, save_path_txt_s)
    save_path_png = os.path.join(vis_vel_dir, f'velocity_t{fluid.timestep:03d}.png')
    save_figure(fig, save_path_png)

    save_path_txt_v = os.path.join(txt_dir, f'vorticity_values_t{fluid.timestep:03d}.txt')
    save_path_txt_s = os.path.join(txt_dir, f'vorticity_samples_t{fluid.timestep:03d}.txt')
    fig = fluid.draw_vorticity(cfg.vis_resolution, save_path_txt_v, save_path_txt_s)
    save_path = os.path.join(vis_vor_dir, f'vorticity_t{fluid.timestep:03d}.png')
    save_figure(fig, save_path)

    # Plot kinetic energy
    # E_k = fluid.compute_kinetic_energy(cfg.vis_resolution)
    # energy.append(E_k)
    # timestep.append(fluid.timestep)
    # plt.plot(timestep, energy)
    # save_path = os.path.join(cfg.results_dir, 'energy.png')
    # plt.savefig(save_path)

    fluid.save_ckpt()
    # save_path = os.path.join(cfg.results_dir, 'energy.txt')
    # np.savetxt(save_path, energy)

    plt.close("all")
