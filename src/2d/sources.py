import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import gpytoolbox

def get_source_velocity(src, i = 1):
    if src == 'taylorgreen':
        source_func = taylorgreen_velocity
    elif src == 'karman':
        source_func = karman_vortex_velocity
    elif src == 'jpipe':
        source_func = jpipe_velocity
    else:
        raise NotImplementedError
    return source_func

def taylorgreen_velocity(samples: torch.FloatTensor, scene_size=None):
    # samples: [-1, 1], rescale to (0, 2 * pi)
    A = 1
    a = 1
    B = -1
    b = 1
    x = ((samples[:, 0]-scene_size[0])/(scene_size[1] - scene_size[0])) * 2*np.pi
    y = ((samples[:, 1]-scene_size[2])/(scene_size[3] - scene_size[2])) * 2*np.pi
    u = A * torch.sin(a * x) * torch.cos(b * y)
    v = B * torch.cos(a * x) * torch.sin(b * y)
    vel = torch.stack([u, v], dim=-1)
    
    return vel

def karman_vortex_velocity(samples: torch.FloatTensor, karman_vel, obs_func, scene_size, eps):
    vel = torch.zeros_like(samples)
    vel[..., 0] = karman_vel

    dist = obs_func(samples)
    threshold = eps
    weight = torch.clamp(dist, 0, threshold) / threshold
    vel *= weight.unsqueeze(-1)

    return vel

def jpipe_velocity(samples: torch.FloatTensor, karman_vel, obs_func, eps):
    vel = torch.zeros_like(samples)
    mask = samples[..., 0] < 1.4
    vel[..., 0][mask] = karman_vel

    dist = obs_func(samples)
    threshold = eps
    weight = torch.clamp(dist, 0, threshold) / threshold
    vel *= weight.unsqueeze(-1)

    d = torch.sqrt((samples[..., 0]-1)**2 + (samples[..., 1]-1)**2)
    mask1 = (samples[..., 0]>=0.0) & (samples[..., 0]<=1.0) & (samples[..., 1]>=0.0) & (samples[..., 1]<=0.5)
    mask2 = (samples[..., 0]>=1.5) & (samples[..., 0]<=2.0) & (samples[..., 1]>=1.0) & (samples[..., 1]<=2.0)
    mask3 = (d >= 0.5) & (d <= 1.0) & (samples[..., 0]>=1.0) & (samples[..., 1]<=1.0)
    mask = mask1 | mask2 | mask3
    vel[~mask] = 0.0

    # dist = obs_func(samples)
    # threshold = eps
    # weight = torch.clamp(dist, 0, threshold) / threshold
    # vel *= weight.unsqueeze(-1)

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

def circle_obstable_functions(center, radius):
    def sdf_func(samples):
        d = torch.sqrt((samples[..., 0] - center[0]) ** 2 + (samples[..., 1] - center[1]) ** 2) - radius
        return d
    
    return sdf_func

def jpipe_obstable_functions():
    def sdf_func(samples):
        dist = torch.zeros_like(samples[..., 0])

        mask1 = (samples[..., 0] >= 0.0) & (samples[..., 0] <= 1.0)
        mask2 = (samples[..., 1] >= 1.0) & (samples[..., 1] <= 2.0)
        mask = ~mask1 & ~mask2

        dist[mask1] = torch.minimum(torch.abs(samples[..., 1][mask1] - 0.5), torch.abs(samples[..., 1][mask1]))
        dist[mask2] = torch.minimum(torch.abs(samples[..., 0][mask2] - 1.5), torch.abs(samples[..., 0][mask2] - 2.0))
        dist[mask] = torch.minimum(torch.abs(torch.sqrt((samples[..., 0][mask]-1)**2 + (samples[..., 1][mask]-1)**2) - 0.5), torch.abs(torch.sqrt((samples[..., 0][mask]-1)**2 + (samples[..., 1][mask]-1)**2) - 1))

        return dist
    return sdf_func

def obstacle_function(v, l):
    def sdf_func(samples):
        samples = samples.detach().cpu().numpy()
        dist = np.zeros(samples[..., 0].shape)
        if len(samples.shape) == 3:
            for i in range(samples.shape[0]):
                winding_number = gpytoolbox.winding_number(samples[i], v, l)
                d, _, _ = gpytoolbox.signed_distance(samples[i], v, F=l, use_cpp=True)
                d = d * np.sign(winding_number)
                dist[i] = d
        else:
            winding_number = gpytoolbox.winding_number(samples, v, l)
            d, _, _ = gpytoolbox.signed_distance(samples, v, F=l, use_cpp=True)
            d = d * np.sign(winding_number)
            dist = d
        return torch.Tensor(dist).cuda()
        
    return sdf_func
