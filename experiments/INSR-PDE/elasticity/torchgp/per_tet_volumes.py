import torch

def per_tet_volumes(
    V : torch.Tensor,
    T : torch.Tensor):
    """Compute normals per face.

    Args:
        mesh (torch.Tensor): #F, 4, 3 array of vertices
    """

    vec_a = V[T[:, 1], :] - V[T[:, 0], :]
    vec_b = V[T[:, 2], :] - V[T[:, 0], :]
    vec_c = V[T[:, 3], :] - V[T[:, 0], :]
    normals = torch.cross(vec_a, vec_b)

    volumes = torch.abs(torch.sum(vec_c * normals, dim=-1)) / 6
    return volumes
