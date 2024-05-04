import torch
from .per_face_normals import per_face_normals

def per_face_areas(
    V : torch.Tensor,
    F : torch.Tensor):
    """Compute normals per face.

    Args:
        mesh (torch.Tensor): #F, 4, 3 array of vertices
    """

    normals = per_face_normals(V, F)
    areas = torch.norm(normals, p=2, dim=1) * 0.5
    return areas
