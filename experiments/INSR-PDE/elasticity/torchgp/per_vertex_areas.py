from .per_face_areas import per_face_areas
from .per_tet_volumes import per_tet_volumes
import torch

def per_vertex_areas(
    V : torch.Tensor,
    T : torch.Tensor):
    """Compute barycentric area per vertex.

    Args:
        mesh (torch.Tensor): #F, 4, 3 array of vertices
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if T.shape[1] == 4:
      volumes = per_tet_volumes(V, T)
      i0 = T[:, 0]
      i1 = T[:, 1]
      i2 = T[:, 2]
      i3 = T[:, 3]
      i = torch.cat((i0,i1,i2,i3), dim=0) - 1
      volumes_tmp = volumes / 4.
      volumes_bary = torch.cat((volumes_tmp, volumes_tmp, volumes_tmp, volumes_tmp), dim=0)
      volumes_per_vert = torch.zeros(V.shape[0], 1, device = device)
      volumes_per_vert[i, :] += volumes_bary[..., None]
      return volumes_per_vert
    elif T.shape[1] == 3:
      areas = per_face_areas(V, T)
      i0 = T[:, 0]
      i1 = T[:, 1]
      i2 = T[:, 2]
      i = torch.cat((i0,i1,i2), dim=0)
      areas_tmp = areas / 3.
      areas_bary = torch.cat((areas_tmp, areas_tmp, areas_tmp), dim=0)
      areas_per_vert = torch.zeros(V.shape[1], 1)
      areas_per_vert[i, :] += areas_bary
      return areas_per_vert
    else:
      raise NotImplementedError


  
