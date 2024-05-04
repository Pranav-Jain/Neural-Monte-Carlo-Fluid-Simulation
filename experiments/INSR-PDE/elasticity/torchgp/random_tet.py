import torch
from .volume_weighted_distribution import volume_weighted_distribution
from .per_tet_volumes import per_tet_volumes

def random_tet(
    V : torch.Tensor, 
    T : torch.Tensor, 
    num_samples : int, 
    distrib=None):
    """Return an volume weighted random sample of faces and their normals from the mesh.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        T (torch.Tensor): #T, 4 array of indices
        num_samples (int): num of samples to return
        distrib: distribution to use. By default, volume-weighted distribution is used.
    """
    if distrib is None:
        distrib = volume_weighted_distribution(V, T)

    idx = distrib.sample([num_samples])

    return T[idx, :]