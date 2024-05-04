import torch
import torch.nn.functional as F
from functools import partial
import numpy as np
import math
import os

def get_source_velocity(src, i = 1):
    if src == 'example1':
        source_func = example1_velocity
    elif src == 'example2':
        source_func = example2_velocity
    elif src == 'example3':
        source_func = example3_velocity
    elif src == 'example4':
        source_func = example4_velocity
    elif src == 'example5':
        source_func = example5_velocity
    elif src == 'example6':
        source_func = example6_velocity
    elif src == 'taylorgreen':
        source_func = partial(taylorgreen_velocity)
    elif src == 'taylorgreen_multi':
        source_func = taylorgreen_multi_velocity
    elif src == 'vortexsheet':
        source_func = vortexsheet_velocity
    elif src == 'vortexsheet_blend':
        source_func = vortexsheet_velocity_blend
    elif os.path.exists(src):
        source_func = load_from_discrete_velocity(src, i)  # use first frame (after one step) by default
    else:
        raise NotImplementedError
    return source_func


def get_source_density(src):
    if src == 'vortexsheet_blend':
        source_func = vortexsheet_density_blend
    elif src == 'taylorgreen_multi':
        source_func = taylorgreen_multi_density
    elif src == 'taylorgreen':
        source_func = taylorgreen_density
    else:
        raise NotImplementedError
    return source_func


def get_source_pressure(src):
    if src == 'taylorgreen':
        source_func = taylorgreen_pressure
    else:
        raise NotImplementedError

    return source_func


def example1_velocity(samples: torch.FloatTensor, radius=0.25, density=1):
    return torch.zeros_like(samples)


def normal_pdf(x, mu, sigma):
    return 1. / (sigma * math.sqrt(2 * math.pi)) * torch.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))


def example2_density(samples: torch.FloatTensor, mu=0, sigma=0.1, density=1):
    dists = torch.sqrt(torch.sum(samples ** 2, dim=-1, keepdim=True))
    return normal_pdf(dists, mu, sigma) * density


def example2_velocity(samples: torch.FloatTensor, mu=0, sigma=0.25, velocity=1):
    dists = torch.sqrt(torch.sum(samples ** 2, dim=-1, keepdim=True))
    dirc = samples / dists # normalized direction vector
    vel = dirc * normal_pdf(dists, mu, sigma) * velocity
    return vel


def example3_velocity(samples: torch.FloatTensor, mu=0, sigma=0.25, velocity=10):
    dists = torch.sqrt(torch.sum(samples ** 2, dim=-1, keepdim=True))
    dirc = torch.zeros([1] * (len(samples.shape) - 1) + [2], dtype=samples.dtype, device=samples.device)
    dirc[..., 0] = 1
    vel = dirc * normal_pdf(dists, mu, sigma) * velocity
    return vel
    

def tukey_window1D(samples, halfL, alpha=0.5):
    """samples: (...,)"""
    samples_ref = torch.where(samples > 0, -samples, samples) + halfL
    out = torch.ones_like(samples)
    mask1 = torch.logical_and(samples_ref < alpha * halfL, samples_ref > 0)
    out[mask1] = 0.5 * (1 - torch.cos(2 * torch.pi * samples_ref[mask1] / (alpha * halfL * 2)))
    mask2 = samples_ref <= 0
    out[mask2] = 0
    return out


def example4_velocity(samples: torch.FloatTensor, halfL=0.5, alpha=0.5, velocity=5):
    """samples (..., 2)"""
    val = tukey_window1D(samples[..., 0], halfL, alpha) * tukey_window1D(samples[..., 1], halfL, alpha)
    dirc = torch.zeros([1] * (len(samples.shape) - 1) + [2], dtype=samples.dtype, device=samples.device)
    dirc[..., 0] = 1
    vel = dirc * val.unsqueeze(-1) * velocity
    vel[vel < 1e-3] = 0
    return vel


def example5_velocity(samples: torch.FloatTensor, center=(-0.5, 0), mu=0, sigma=0.15, velocity=1):
    center_ = torch.zeros([1] * (len(samples.shape) - 1) + [2], dtype=samples.dtype, device=samples.device)
    center_[..., 0] = center[0]
    center_[..., 1] = center[1]

    dists = torch.sqrt(torch.sum((samples - center_) ** 2, dim=-1, keepdim=True))
    dirc = torch.zeros([1] * (len(samples.shape) - 1) + [2], dtype=samples.dtype, device=samples.device)
    dirc[..., 0] = 1
    vel = dirc * normal_pdf(dists, mu, sigma) * velocity
    return vel


def example6_velocity(samples: torch.FloatTensor, center1=(-0.5,0), center2=(0.5,0), r1=0.4, r2=0.2, blend_offset=0.02, velocity=1):
    # one large vortex and one small vortex
    w1 = 5.0
    w2 = 10.0
    c1 = torch.Tensor([center1[0], center1[1]]).cuda()
    c2 = torch.Tensor([center2[0], center2[1]]).cuda()
    samples_ref1 = torch.norm(samples - c1, dim=-1)
    samples_ref2 = torch.norm(samples - c2, dim=-1)
    samples_blend_ratio1 = torch.clamp((r1 - samples_ref1) / blend_offset, min=0.0, max=1.0)
    samples_blend_ratio2 = torch.clamp((r2 - samples_ref2) / blend_offset, min=0.0, max=1.0)
    u1 = - w1 * (samples[..., 0] - center1[0]) * samples_blend_ratio1
    v1 = w1 * (samples[..., 1] - center1[1]) * samples_blend_ratio1
    mask1 = (samples_ref1 < r1)
    u1[~mask1] = 0
    v1[~mask1] = 0
    u2 = - w2 * (samples[..., 0] - center2[0]) * samples_blend_ratio2
    v2 = w2 * (samples[..., 1] - center2[1]) * samples_blend_ratio2
    mask2 = (samples_ref2 < r2)
    u2[~mask2] = 0
    v2[~mask2] = 0
    u = u1 + u2
    v = v1 + v2
    vel = torch.stack([v, u], dim=-1)
    return vel


def taylorgreen_velocity(samples: torch.FloatTensor, rescale=False):
    # samples: [-1, 1], rescale to (0, 2 * pi)
    A = 1
    a = 1
    B = -1
    b = 1
    x = (samples[..., 0] + 1) * math.pi
    y = (samples[..., 1] + 1) * math.pi
    u = A * torch.sin(a * x) * torch.cos(b * y)
    v = B * torch.cos(a * x) * torch.sin(b * y)
    # if rescale:
    #     u = u / math.pi
    #     v = v / math.pi
    vel = torch.stack([u, v], dim=-1)
    return vel


def taylorgreen_pressure(samples: torch.FloatTensor, rho=1.):
    # samples: [-1, 1], rescale to (0, 2 * pi)
    samples_rescale = (samples + 1.) * np.pi
    p = rho / 4. * (torch.cos(2 * samples_rescale[..., 0]) + torch.sin(2 * samples_rescale[... , 1]))
    return p


def taylorgreen_density(samples: torch.FloatTensor, rigidR=0.5):
    samples_ref = torch.norm(samples, p=2, dim=-1)
    den = torch.zeros(samples.shape[:2])
    den[samples_ref < rigidR] = 1.0
    return den


tlgnM_scale = 8

def taylorgreen_multi_velocity(samples: torch.FloatTensor):
    # samples: [-1, 1], rescale to (0, 2 * pi)
    gap = 0.05 / tlgnM_scale * 8

    vel = torch.zeros_like(samples)
    # [-1, 0] x [-1, 0]
    mask = torch.logical_and(samples[..., 0] <= 0 + gap, samples[..., 1] <= 0 + gap)
    weight = 1. - (samples[mask] - torch.tensor([[0, 0]]).to(samples)).clamp(min=0, max=gap).norm(dim=-1) / gap
    vel[mask] = taylorgreen_velocity(torch.clamp(samples[mask] * 2 + 1, min=-1, max=1)) * weight.unsqueeze(-1)
    # vel[mask] *= 1 / 2

    # (0.5, 1] x (0.5, 1]
    p = 1 - 2 / tlgnM_scale
    gap_ = gap * 2 / tlgnM_scale
    mask = torch.logical_and(samples[..., 0] > p - gap_, samples[..., 1] > p - gap_)
    weight = 1. - (torch.tensor([[p, p]]).to(samples) - samples[mask]).clamp(min=0, max=gap_).norm(dim=-1) / gap_
    vel[mask] = taylorgreen_velocity(torch.clamp(samples[mask] * tlgnM_scale + torch.tensor([[-tlgnM_scale + 1, -tlgnM_scale + 1]]).to(samples), min=-1, max=1)) * weight.unsqueeze(-1)
    # vel[mask] *= 2 / tlgnM_scale

    return vel


def taylorgreen_multi_density(samples: torch.FloatTensor):
    den = torch.zeros_like(samples)[..., 0]

    mask = torch.logical_and(samples[..., 0] <= 0, samples[..., 1] <= 0)
    den[mask] = vortexsheet_density(samples[mask] * 2 + 1)

    mask = torch.logical_and(samples[..., 0] > 1 - 2 / tlgnM_scale, samples[..., 1] > 1 - 2 / tlgnM_scale)
    den[mask] = vortexsheet_density(samples[mask] * tlgnM_scale + torch.tensor([[-tlgnM_scale + 1, -tlgnM_scale + 1]]).to(samples))
    return den


def vortexsheet_velocity(samples: torch.FloatTensor, rigidR=0.4, rate=0.1):
    w = 1 * 1.0 / rate
    samples_ref = torch.norm(samples, dim=-1)
    mask1 = (samples_ref < rigidR)
    u = w * samples[..., 1]
    v = -w * samples[..., 0]
    u[~mask1] = 0
    v[~mask1] = 0
    vel = torch.stack([u, v], dim=-1)
    return vel
    

def vortexsheet_velocity_blend(samples: torch.FloatTensor, rigidR=0.5, blend_offset=0.1, rate=0.1):
    ## Set the value at rigidR to be 0 and blend between [rigidR - blend_offset, rigidR]
    w = 1 * 1.0 / rate
    samples_ref = torch.norm(samples, p=2, dim=-1)
    # samples_blend_ratio = torch.clamp((rigidR - samples_ref) / blend_offset, min=0.0, max=1.0)
    samples_blend_ratio = tukey_window1D(samples_ref, 0.5, 0.3)
    u = w * samples[..., 1] * samples_blend_ratio
    v = -w * samples[..., 0] * samples_blend_ratio
    # mask1 = (samples_ref < rigidR)
    # u[~mask1] = 0
    # v[~mask1] = 0
    vel = torch.stack([u, v], dim=-1)
    return vel


def vortexsheet_density(samples: torch.FloatTensor, rigidR=0.5):
    samples_ref = torch.norm(samples, p=2, dim=-1)
    den = torch.zeros_like(samples_ref)
    den[samples_ref < rigidR] = 1.0
    return den


def vortexsheet_density_blend(samples: torch.FloatTensor, rigidR=0.5):
    samples_ref = torch.norm(samples, p=2, dim=-1)
    # den = torch.zeros(samples.shape[:2])
    # den[samples_ref < rigidR] = 1.0
    den = tukey_window1D(samples_ref, 0.5, 0.3)
    return den


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
