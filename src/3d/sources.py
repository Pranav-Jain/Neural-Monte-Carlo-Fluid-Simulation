import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import gpytoolbox

def get_source_velocity(src, i = 1):
    if src == 'smoke':
        source_func = smoke_velocity
    elif src == 'vortex_collide':
        source_func = vortex_collide
    elif src == 'karman3d':
        source_func = karman_velocity
    elif src == 'smoke_obs':
        source_func = smoke_obs_velocity
    else:
        raise NotImplementedError
    return source_func


def smoke_velocity(samples: torch.FloatTensor):
    # e = torch.Tensor([1.0, 0.0], device=samples.device)
    vel = torch.zeros_like(samples)

    center1 = torch.Tensor([0.0, 0.0, -0.6])
    radius1 = 0.11
    # dir1 = torch.stack((samples[:, 0] - center1[0], samples[:, 1] - center1[1]), dim=-1)
    # dir1 = dir1/torch.linalg.norm(dir1, axis=0)
    # theta1 = torch.acos(dir1[:, 0])
    dist = torch.linalg.norm(samples - center1.cuda(), axis=-1)
    mask1 = dist < radius1

    r = torch.Tensor(np.random.random(vel[mask1].shape[0])).cuda()
    r = r*2 - 1
    r *= 10
    vel[..., 0][mask1] = 0.01*r
    vel[..., 1][mask1] = 0.01*r
    vel[..., 2][mask1] = 0.2+0.01*r

    # center2 = torch.Tensor([0.201, 0.2, -0.4])
    # radius2 = 0.2
    # dir2 = torch.stack((samples[:, 0] - center2[0], samples[:, 1] - center2[1]), dim=-1)
    # dir2 = dir2/torch.linalg.norm(dir2, axis=0)
    # theta2 = torch.acos(dir2[:, 0])
    # mask2 = (torch.linalg.norm(samples.cpu() - center2, axis=1) < radius2)
    # vel[:, 2][mask2] = 0.06 * (1.+0.01*torch.cos(8*theta2[mask2]))

    return vel

def smoke_obs_velocity(samples: torch.FloatTensor):
    # e = torch.Tensor([1.0, 0.0], device=samples.device)
    vel = torch.zeros_like(samples)

    center1 = torch.Tensor([0.0, 0.0, -0.6])
    radius1 = 0.11
    dist = torch.linalg.norm(samples - center1.cuda(), axis=-1)
    mask1 = dist < radius1
    
    # r = torch.Tensor(np.random.random(vel[mask1].shape[0])).cuda()
    # r = r*2 - 1
    # r *= 3
    # vel[..., 0][mask1] = 0.01*r
    # vel[..., 1][mask1] = 0.01*r
    # vel[..., 2][mask1] = 0.2+0.01*r
    vel[..., 2][mask1] = 1.0

    return vel

def vortex_collide(samples: torch.FloatTensor):
    samples = samples.cuda()
    vel = torch.zeros_like(samples)
    center1 = torch.Tensor([0.0, 0.0, -0.21])
    radius1 = 0.2
    dir1 = torch.stack((samples[:, 0] - 0.2, samples[:, 1] - 0.2), dim=-1)
    dir1 = dir1/torch.linalg.norm(dir1, axis=0)
    theta1 = torch.acos(dir1[:, 0])
    mask1 = (torch.linalg.norm(samples.cpu() - center1, axis=1) < radius1)
    # mask1 = (torch.sqrt(samples[..., 0]**2 + samples[..., 1]**2) < radius1) & (samples[..., 2]>-0.08) & (samples[..., 2]<-0.02)
    vel[:, 2][mask1] = 0.2 * (1.+0.01*torch.cos(8*theta1[mask1]))
    # vel[:, 2][mask1] = -1

    center2 = torch.Tensor([0.0, 0.0, 0.21])
    radius2 = 0.2
    dir2 = torch.stack((samples[:, 0] - 0.201, samples[:, 1] - 0.2), dim=-1)
    dir2 = dir2/torch.linalg.norm(dir2, axis=0)
    theta2 = torch.acos(dir2[:, 0])
    mask2 = (torch.linalg.norm(samples.cpu() - center2, axis=1) < radius2)
    # mask2 = (torch.sqrt(samples[..., 0]**2 + samples[..., 1]**2) < radius1) & (samples[..., 2]<0.08) & (samples[..., 2]>0.02)
    vel[:, 2][mask2] = -0.2 * (1.+0.01*torch.cos(8*theta2[mask2]))
    # vel[:, 2][mask2] = 1

    return vel

def karman_velocity(samples: torch.FloatTensor, karman_vel, obs_func, eps):
    vel = torch.zeros_like(samples)
    vel[..., 2] = karman_vel

    dist = obs_func(samples)
    threshold = eps
    weight = torch.clamp(dist, 0, threshold) / threshold
    vel *= weight.unsqueeze(-1)

    return vel


def _smoothstep_linear(x, xm, e):
        y = torch.abs(x-xm)
        return y/e

def _smoothstep_poly(x, xm, e):
        y = torch.abs(x-xm)
        return y + (((3-2*e)/e**2) * y**2) + (((e-2)/e**3) * y**3)

def _smoothstep_tanh(x, xm, e):
        y = torch.abs(x-xm)
        return ((torch.exp(y) - torch.exp(-y))/(torch.exp(y) + torch.exp(-y))) * ((np.exp(e) + np.exp(-e))/(np.exp(e) - np.exp(-e)))

def load_from_discrete_velocity(path, i=1):
    value_grid = np.load(path)[i] # use first frame (after one step) by default
    value_grid = torch.from_numpy(value_grid).float().permute(2, 0, 1).unsqueeze(0).cuda()

    def interpolate(samples: torch.FloatTensor):
        if len(samples.shape) == 3:
            # FIXME: switch xy is weired. problem?
            vel = F.grid_sample(value_grid, samples[..., [1, 0]].unsqueeze(0), 
                mode='bilinear', padding_mode="zeros", align_corners=False).squeeze(0).permute(1, 2, 0)
        else:
            vel = F.grid_sample(value_grid, samples[..., [1, 0]].unsqueeze(0).unsqueeze(0), 
                mode='bilinear', padding_mode="zeros", align_corners=False).squeeze(0).permute(1, 2, 0).squeeze(0)
        return vel
    return interpolate

def circle_obstable_functions(center, radius):
    def sdf_func(samples):
        dist = torch.sqrt((samples[..., 0] - center[0]) ** 2 + (samples[..., 1] - center[1]) ** 2 + (samples[..., 2] - center[2]) ** 2) - radius
        return dist
    
    return sdf_func

def cylinder_obstacle_function(center, radius):
    def sdf_func(samples):
        dist = torch.sqrt((samples[..., 0]-center[0])**2 + (samples[..., 2]-center[1])**2) - radius
        return dist
    return sdf_func