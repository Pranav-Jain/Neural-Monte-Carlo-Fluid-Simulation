from matplotlib.pyplot import axis
import torch
import numpy as np
from .random_tet import random_tet
from .volume_weighted_distribution import volume_weighted_distribution

import time

def sample_volume(
    V : torch.Tensor,
    T : torch.Tensor,
    num_samples : int,
    distrib = None):
    """Sample points and their normals on mesh volume.

    Args:
        V (torch.Tensor): #V, 3 array of vertices
        T (torch.Tensor): #T, 4 array of indices
        num_samples (int): number of volume samples
        distrib: distribution to use. By default, volume-weighted distribution is used
    """
    sample_volume_start_time = time.time()

    dist_start_time = time.time()
    if distrib is None:
        distrib = volume_weighted_distribution(V, T)
    dist_time = time.time() - dist_start_time

    rand_start_time = time.time()
    # Select tets & sample their volume
    tidx = random_tet(V, T, num_samples, distrib)
    tet = V[tidx]
    rand_time = time.time() - rand_start_time

    bary_start_time = time.time()
    ## Method 1: Generate random coordinate using dirichlet distribution
    barys = np.random.dirichlet((1.0,1.0,1.0,1.0), size = (num_samples, 1))
    barys = torch.from_numpy(barys.squeeze().astype(np.float32)).to(V.device)

    samples = barys[:,0,None] * tet[:,0,:] + barys[:,1,None] * tet[:,1,:] + barys[:,2,None] * tet[:,2,:] + barys[:,3,None] * tet[:,3,:]
    bary_time = time.time() - bary_start_time
    
    check_start_time = time.time()
    assert(~torch.any(torch.isnan(samples)))
    check_time = time.time() - check_start_time

    sample_volume_time = time.time() - sample_volume_start_time

    # print("sample dist time percentage = %f" % (dist_time / sample_volume_time))
    # print("sample rand tet time percentage = %f" % (rand_time / sample_volume_time))
    # print("sample bary time percentage = %f" % (bary_time / sample_volume_time))
    # print("sample check time percentage = %f" % (check_time / sample_volume_time))

    # # Method 2: Generate random barycentric coordinates inside the tetrahedron
    # # It turns out that it will generate negative coefficients for barycentric coordinates :( So bad. 
    # # Reference: http://vcg.isti.cnr.it/publications/papers/rndtetra_a.pdf
    # s = torch.rand(num_samples).to(V.device).unsqueeze(-1)
    # t = torch.rand(num_samples).to(V.device).unsqueeze(-1)
    # u = torch.rand(num_samples).to(V.device).unsqueeze(-1)

    # stu = s + t + u
    # tu = t + u

    # idx_stu = torch.squeeze(stu <= 1.0)
    # idx_tu = torch.squeeze(tu > 1)
    # idx_other = torch.squeeze((stu > 1.0) & (tu <= 1.0))

    # barys = torch.zeros((num_samples, 4)).to(V.device)
    # barys[idx_stu, 0:3] = torch.hstack((s[idx_stu], t[idx_stu], u[idx_stu]))
    # barys[idx_tu, 0:3] = torch.hstack((s[idx_tu], 1.0-u[idx_tu], 1.0-s[idx_tu]-t[idx_tu]))
    # barys[idx_other, 0:3] = torch.hstack((1.0-t[idx_other]-u[idx_other], t[idx_other], s[idx_other]+t[idx_other]+u[idx_other]-1.0))
    
    # barys[:, 3] = 1.0 - torch.sum(barys[:, 0:3], axis = 1)
    
    return samples