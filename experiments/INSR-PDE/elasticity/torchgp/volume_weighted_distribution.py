import torch
from .per_tet_volumes import per_tet_volumes

def volume_weighted_distribution(
    V : torch.Tensor,
    T : torch.Tensor, 
    volumes : torch.Tensor = None):
    """Construct discrete area weighted distribution over triangle mesh.

    Args:
        mesh (torch.Tensor): #F, 3, 3 array of vertices
        normals (torch.Tensor): normals (if precomputed)
        eps (float): epsilon
    """

    if volumes is None:
        volumes = per_tet_volumes(V, T)

    volumes /= torch.sum(volumes) + 1e-10

    assert all(vol > 0 for vol in volumes)
    # Discrete PDF over tetrahedrons
    return torch.distributions.Categorical(volumes.view(-1))