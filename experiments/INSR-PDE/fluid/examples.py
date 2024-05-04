import math
import torch
from functools import partial


def get_examples(src, **kwargs):
    # select source function
    if src == 'taylorgreen':
        source_func = partial(taylorgreen_velocity)
    elif src == 'taylorgreen_multi':
        source_func = taylorgreen_multi_velocity
    else:
        raise NotImplementedError
    return source_func


def taylorgreen_velocity(samples: torch.FloatTensor, rescale=False):
    # samples: [-1, 1]
    A = 1
    a = 1
    B = -1
    b = 1
    x = (samples[..., 0] + 1) * math.pi
    y = (samples[..., 1] + 1) * math.pi
    u = A * torch.sin(a * x) * torch.cos(b * y)
    v = B * torch.cos(a * x) * torch.sin(b * y)
    if rescale:
        u = u / math.pi
        v = v / math.pi
    vel = torch.stack([u, v], dim=-1)
    return vel


def taylorgreen_multi_velocity(samples: torch.FloatTensor, scale=8):
    # samples: [-1, 1]
    gap = 0.05

    vel = torch.zeros_like(samples)
    # [-1, 0] x [-1, 0]
    mask = torch.logical_and(samples[..., 0] <= 0 + gap, samples[..., 1] <= 0 + gap)
    weight = 1. - (samples[mask] - torch.tensor([[0, 0]]).to(samples)).clamp(min=0, max=gap).norm(dim=-1) / gap
    vel[mask] = taylorgreen_velocity(torch.clamp(samples[mask] * 2 + 1, min=-1, max=1)) * weight.unsqueeze(-1)

    # (0.5, 1] x (0.5, 1]
    p = 1 - 2 / scale
    gap_ = gap * 2 / scale
    mask = torch.logical_and(samples[..., 0] > p - gap_, samples[..., 1] > p - gap_)
    weight = 1. - (torch.tensor([[p, p]]).to(samples) - samples[mask]).clamp(min=0, max=gap_).norm(dim=-1) / gap_
    vel[mask] = taylorgreen_velocity(torch.clamp(samples[mask] * scale + torch.tensor([[-scale + 1, -scale + 1]]).to(samples), min=-1, max=1)) * weight.unsqueeze(-1)

    return vel
