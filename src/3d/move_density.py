import os
from tqdm import tqdm
import torch
import numpy as np
from config import Config
from models import get_model
from functools import partial
from utils.vis_utils import draw_scalar_field2D, save_figure, frames2gif
from scipy.ndimage import map_coordinates
import gpytoolbox
import matplotlib.pyplot as plt
import pyopenvdb as vdb
from numba import jit

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
    v, f = gpytoolbox.read_mesh(filename)
    min_x = np.min(v[:, 0])
    max_x = np.max(v[:, 0])
    min_y = np.min(v[:, 1])
    max_y = np.max(v[:, 1])
    min_z = np.min(v[:, 2])
    max_z = np.max(v[:, 2])

    # scene_size = [max_x - min_x, max_y - min_y]
    scene_size = [min_x, max_x, min_y, max_y, min_z, max_z]
    
    return scene_size

def smoke(samples):
    vel = torch.zeros_like(samples).float()

    center1 = torch.Tensor([0.0, 0.0, -0.6])
    radius1 = 0.11
    # dir1 = torch.stack((samples[:, 0] - center1[0], samples[:, 1] - center1[1]), dim=-1)
    # dir1 = dir1/torch.linalg.norm(dir1, axis=0)
    # theta1 = torch.acos(dir1[:, 0])
    dist = torch.linalg.norm(samples - center1, axis=-1)
    mask1 = dist < radius1

    r = torch.Tensor(np.random.random(vel[mask1].shape[0]))
    r = r*2 - 1
    r *= 10
    vel[..., 0][mask1] = 0.01*r
    vel[..., 1][mask1] = 0.01*r
    vel[..., 2][mask1] = 0.2+0.01*r

    out = np.linalg.norm(vel, axis=-1)
    return out

def smoke_obs(samples: torch.FloatTensor):
    # e = torch.Tensor([1.0, 0.0], device=samples.device)
    vel = torch.zeros_like(samples)

    center1 = torch.Tensor([0.0, 0.0, -0.6])
    radius1 = 0.11
    dist = torch.linalg.norm(samples - center1, axis=-1)
    mask1 = dist < radius1
    
    # r = torch.Tensor(np.random.random(vel[mask1].shape[0])).cuda()
    # r = r*2 - 1
    # r *= 3
    # vel[..., 0][mask1] = 0.01*r
    # vel[..., 1][mask1] = 0.01*r
    # vel[..., 2][mask1] = 0.2+0.01*r
    vel[..., 2][mask1] = 1.0

    out = np.linalg.norm(vel, axis=-1)
    return out

def vortex_collide(samples: torch.FloatTensor):
    vel = torch.zeros_like(samples)
    
    center1 = torch.Tensor([0.0, 0.0, -0.21])
    radius1 = 0.2
    dir1 = torch.stack((samples[..., 0] - 0.2, samples[..., 1] - 0.2), dim=-1)
    dir1 = dir1/torch.linalg.norm(dir1, axis=0)
    theta1 = torch.acos(dir1[..., 0])
    mask1 = (torch.linalg.norm(samples.cpu() - center1, axis=-1) < radius1)
    vel[..., 2][mask1] = 0.2 * (1.+0.01*torch.cos(8*theta1[mask1]))
    # vel[:, 2][mask1] = -1

    center2 = torch.Tensor([0.0, 0.0, 0.21])
    radius2 = 0.2
    dir2 = torch.stack((samples[..., 0] - 0.2, samples[..., 1] - 0.2), dim=-1)
    dir2 = dir2/torch.linalg.norm(dir2, axis=0)
    theta2 = torch.acos(dir2[..., 0])
    mask2 = (torch.linalg.norm(samples.cpu() - center2, axis=-1) < radius2)
    vel[..., 2][mask2] = -0.2 * (1.+0.01*torch.cos(8*theta2[mask2]))
    # vel[:, 2][mask2] = 1

    out = np.linalg.norm(vel, axis=-1)
    col_arr = np.zeros((samples.shape[0], samples.shape[1], samples.shape[2], 3))
    col_arr[..., 0][mask1] = 1.0 # for red
    col_arr[..., 2][mask2] = 1.0 # for blue

    return out, col_arr

def circle_obstable_functions(samples, center=None, radius=None):
    center = torch.Tensor([0.0, -0.5, -0.5])
    radius = 0.2
    dist = torch.sqrt((samples[..., 0] - center[0]) ** 2 + (samples[..., 1] - center[1]) ** 2 + (samples[..., 2] - center[2]) ** 2) - radius
    return dist

def karman_velocity(samples):
    vel = torch.zeros_like(samples)
    vel[..., 2] = cfg.karman_vel

    out = np.linalg.norm(vel, axis=-1)
    print(out.shape)
    # print(out)
    return out

def _draw_scalar_field2D(points, arr, savepath, vmin=None, vmax=None, to_array=False, figsize=(3, 3), cmap='bwr', colorbar=True):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    sc = ax.scatter3D(points[..., 0], points[..., 1], points[..., 2], c=arr, alpha=0.02, cmap=cmap, vmin=None, vmax=None)
    fig.tight_layout()
    if colorbar:
        plt.colorbar(sc)
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close("all")

def _draw_vector_field2D(points, vel, savepath, vmin=None, vmax=None, to_array=False, figsize=(3, 3), cmap='bwr', colorbar=True):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    sc = ax.quiver(points[..., 0], points[..., 1], points[..., 2], vel[..., 0], vel[..., 1], vel[..., 2], length=0.1)
    fig.tight_layout()
    if colorbar:
        plt.colorbar(sc)
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close("all")

@jit(fastmath=True)
def get_curl(V, N):
    F = V.transpose((3,0,1,2))
    VxF = np.zeros(F.shape)
    for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
                VxF[0][i,j,k] = 0.5*((F[2][i,(j+1)%N,k]-
                            F[2][i,j-1,k])-(F[1][i,j,(k+1)%N]-F[1][i,j,k-1]))
                VxF[1][i,j,k] = 0.5*((F[0][i,j,(k+1)%N]-
                            F[0][i,j,k-1])-(F[2][(i+1)%N,j,k]-F[2][i-1,j,k]))
                VxF[2][i,j,k] = 0.5*((F[1][(i+1)%N,j,k]-
                            F[1][i-1,j,k])-(F[0][i,(j+1)%N,k]-F[0][i,j-1,k]))

    return VxF.transpose((1,2,3,0))

# create experiment config containing all hyperparameters
cfg = Config()

if cfg.src == 'smoke' or cfg.src == 'smoke_obs' or cfg.src == 'vortex_collide':
    cfg.scene_size = get_scene_size('./cube.obj')
print(cfg.scene_size)

# create network and training agent
fluid = get_model(cfg)

save_dir = os.path.join(cfg.results_dir, f'density')
os.makedirs(save_dir, exist_ok=True)

N = 200
dt = cfg.dt
interpolate = partial(map_coordinates, 
            order=1, prefilter=False, mode='nearest', cval=0)

d_grid = np.zeros((N, N, N))
grid_coords = np.indices(d_grid.shape)
grid_coords = grid_coords.transpose((1, 2, 3, 0)).astype(float)
grid_coords = (grid_coords)/N*(cfg.scene_size[1] - cfg.scene_size[0]) + cfg.scene_size[0]

# grid_coords1 = (grid_coords + 0.5) / N * 2 - 1.0 # normalize to (-1, 1)

if cfg.src == 'smoke':
    d_grid = smoke(torch.from_numpy(grid_coords))
elif cfg.src == 'smoke_obs':
    d_grid = smoke_obs(torch.from_numpy(grid_coords))
elif cfg.src == 'vortex_collide':
    d_grid, col_arr = vortex_collide(torch.from_numpy(grid_coords))
# d_grid = karman_velocity(torch.from_numpy(grid_coords))

# iterate
grid_coords_torch = torch.tensor((grid_coords), dtype=torch.float32).cuda()

print("\nSaving vdb files...")
for t in tqdm(range(cfg.n_timesteps)):
    try:
        fluid.load_ckpt(t)
        with torch.no_grad():
            grid_vel = fluid.velocity_field(grid_coords_torch).detach().cpu().numpy()

        if t > 0:
            i_back = grid_coords[..., 0] - dt * grid_vel[..., 0]
            j_back = grid_coords[..., 1] - dt * grid_vel[..., 1]
            k_back = grid_coords[..., 2] - dt * grid_vel[..., 2]

            pos = (np.stack([i_back, j_back, k_back]))
            back_pos = (np.stack([i_back, j_back, k_back])-cfg.scene_size[0])*N/(cfg.scene_size[1] - cfg.scene_size[0])
            d_grid = interpolate(d_grid, back_pos)

        den = vdb.FloatGrid()
        vel = vdb.Vec3SGrid()
        den.copyFromArray(d_grid)
        vel.copyFromArray(grid_vel)
        den.transform = vdb.createLinearTransform(voxelSize=0.01)
        vel.transform = vdb.createLinearTransform(voxelSize=0.01)
        den.name = 'density'
        vel.name = 'vel'

        if cfg.src == 'vortex_collide':
            col = vdb.Vec3SGrid()
            col.copyFromArray(col_arr)
            col.transform = vdb.createLinearTransform(voxelSize=0.01)
            col.name = 'Cd'
        
        # curl = vdb.Vec3SGrid()
        # curl_arr = get_curl(grid_vel, N)
        # print(curl_arr[np.abs(curl_arr)<1e-4].shape)
        # curl_arr[np.abs(curl_arr)<1e-4] = 0.0
        # curl.copyFromArray(curl_arr)
        
        save_vdb = os.path.join(save_dir, f'density_t{t:03d}.vdb')
        if cfg.src == 'smoke' or cfg.src == 'smoke_obs':
            vdb.write(save_vdb, grids=[den, vel]) #Writing density, velocity and color in same vdb
        elif cfg.src == 'vortex_collide':
            vdb.write(save_vdb, grids=[den, vel, col]) #Writing density, velocity and color in same vdb
    except:
        break

print("Done")