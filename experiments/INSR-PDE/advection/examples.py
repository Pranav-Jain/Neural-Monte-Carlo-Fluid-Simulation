import torch
from functools import partial


def get_examples(src, **kwargs):
    # select source function
    if src == 'example1':
        source_func = partial(gaussian_like, mu=-1.5)
    else:
        raise NotImplementedError
    return source_func


def gaussian_like(x, mu=0, sigma=0.1):
    """normalized gaussian distribution"""
    return torch.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))
