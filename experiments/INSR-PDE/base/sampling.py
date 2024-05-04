import torch


def sample_uniform(resolution, sdim=1, device="cpu", flatten=True):
    """sample uniform grid points in [-1, 1]^sdim. 
    return shape (resolution, resolution, resolution, sdim)."""
    coords = torch.linspace(0.5, resolution - 0.5, resolution, device=device) / resolution * 2 - 1
    coords = torch.stack(torch.meshgrid([coords] * sdim, indexing='ij'), dim=-1)
    if flatten:
        coords = coords.reshape(resolution**sdim, sdim)
    return coords


def sample_random(N, sdim=1, device="cpu"):
    """sample uniformly random points in in [-1, 1]^sdim.
    return shape (N, sdim)."""
    coords = torch.rand(N, sdim, device=device) * 2 - 1
    return coords


def sample_boundary(N, sdim, epsilon=1e-4, device='cpu'):
    """sample boundary points within a small range. NOTE: random samples, not uniform"""
    if sdim == 1:
        coords_left = (torch.rand(N // 2, 1, device=device) * 2 - 1) * epsilon - 1.
        coords_right = (torch.rand(N // 2, 1, device=device) * 2 - 1) * epsilon + 1.
        coords = torch.cat([coords_left, coords_right], dim=0)
    elif sdim == 2:
        boundary_ranges = [[[-1, 1], [-1 - epsilon, -1 + epsilon]],
                           [[-1, 1], [1 - epsilon, 1 + epsilon]],
                           [[-1 - epsilon, -1 + epsilon], [-1, 1]],
                           [[1 - epsilon, 1 + epsilon], [-1, 1]],]
        coords = []
        for bound in boundary_ranges:
            x_b, y_b = bound
            points = torch.empty(N // 4, 2, device=device)
            points[:, 0] = torch.rand(N // 4, device=device) * (x_b[1] - x_b[0]) + x_b[0]
            points[:, 1] = torch.rand(N // 4, device=device) * (y_b[1] - y_b[0]) + y_b[0]
            coords.append(points)
        coords = torch.cat(coords, dim=0)
    else:
        raise NotImplementedError
    return coords


def sample_boundary2D_separate(N, side, epsilon=1e-4, device='cpu'):
    """sample boundary points within a small range. NOTE: random samples, not uniform"""
    if side == 'horizontal':
        boundary_ranges = [[[-1 - epsilon, -1 + epsilon], [-1, 1]],
                            [[1 - epsilon, 1 + epsilon], [-1, 1]],]
    elif side == 'vertical':
        boundary_ranges = [[[-1, 1], [-1 - epsilon, -1 + epsilon]],
                            [[-1, 1], [1 - epsilon, 1 + epsilon]],]
    else:
        raise RuntimeError

    coords = []
    for bound in boundary_ranges:
        x_b, y_b = bound
        points = torch.empty(N // 2, 2, device=device)
        points[:, 0] = torch.rand(N // 2, device=device) * (x_b[1] - x_b[0]) + x_b[0]
        points[:, 1] = torch.rand(N // 2, device=device) * (y_b[1] - y_b[0]) + y_b[0]
        coords.append(points)
    coords = torch.cat(coords, dim=0)
    return coords
