import torch


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


def sample_boundary_separate(N, side, epsilon=1e-4, device='cpu'):
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


def sample_uniform_2D(resolution: int, normalize=True, with_boundary=False, device='cpu'):
    x = torch.linspace(0.5, resolution - 0.5, resolution, device=device)
    y = torch.linspace(0.5, resolution - 0.5, resolution, device=device)
    if with_boundary:
        x = torch.cat([torch.tensor([0.0], device=device), x, torch.tensor([resolution * 1.0], device=device)])
        y = torch.cat([torch.tensor([0.0], device=device), y, torch.tensor([resolution * 1.0], device=device)])
    # coords = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
    coords = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
    if normalize:
        coords = coords / resolution * 2 - 1

    return coords


def sample_random_2D(N: int, normalize=True, resolution: int=None, device='cpu'):
    coords = torch.rand(N, 2, device=device)
    if normalize:
        coords = coords * 2 - 1
    else:
        coords = coords * resolution
    return coords
