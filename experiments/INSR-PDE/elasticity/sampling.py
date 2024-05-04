import torch
from .torchgp import sample_surface, sample_volume

def sample_mesh(V, F, N, distrib = None):
    if F.shape[1] == 3:
        coords = sample_surface(V, F, N, distrib)
    elif F.shape[1] == 4:
        coords = sample_volume(V, F, N, distrib)
    return coords