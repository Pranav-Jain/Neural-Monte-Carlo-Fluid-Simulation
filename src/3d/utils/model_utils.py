import torch

def sample_uniform_2D(resolution: int, normalize=True, with_boundary=False, size=None, device='cpu'):
    if (size[1] - size[0]) < (size[3] - size[2]):
        if (size[1] - size[0]) < (size[5] - size[4]):
            res_x, res_y, res_z = resolution, int(resolution * (size[3] - size[2]) / (size[1] - size[0])), int(resolution * (size[5] - size[4]) / (size[1] - size[0]))
        else:
            res_x, res_y, res_z = int(resolution * (size[1] - size[0]) / (size[5] - size[4])), int(resolution * (size[3] - size[2]) / (size[5] - size[4])), resolution
    else:
        if (size[3] - size[2]) < (size[5] - size[4]):
            res_x, res_y, res_z = int(resolution * (size[1] - size[0]) / (size[3] - size[2])), resolution, int(resolution * (size[5] - size[4]) / (size[3] - size[2]))
        else:
            res_x, res_y, res_z = int(resolution * (size[1] - size[0]) / (size[5] - size[4])), int(resolution * (size[3] - size[2]) / (size[5] - size[4])), resolution

    x = torch.linspace(0.5, res_x - 0.5, res_x, device=device)
    y = torch.linspace(0.5, res_y - 0.5, res_y, device=device)
    z = torch.linspace(0.5, res_z - 0.5, res_y, device=device)
    if with_boundary:
        x = torch.cat([torch.tensor([0.0], device=device), x, torch.tensor([res_x * 1.0], device=device)])
        y = torch.cat([torch.tensor([0.0], device=device), y, torch.tensor([res_y * 1.0], device=device)])
        z = torch.cat([torch.tensor([0.0], device=device), z, torch.tensor([res_z * 1.0], device=device)])
    # coords = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
    coords = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1)
    if normalize:
        coords[..., 0] = coords[..., 0] / res_x * (size[1] - size[0]) + size[0]
        coords[..., 1] = coords[..., 1] / res_y * (size[3] - size[2]) + size[2]
        coords[..., 2] = coords[..., 2] / res_z * (size[5] - size[4]) + size[4]

    return coords

def sample_random_2D(N: int, normalize=True, size=None, device='cpu', epsilon=1e-1, obs_v=None):
    coords = torch.rand(N, 3, device=device)
    if normalize:
        coords[..., 0] = coords[..., 0] * (size[1] - size[0]) + size[0]
        coords[..., 1] = coords[..., 1] * (size[3] - size[2]) + size[2]
        coords[..., 2] = coords[..., 2] * (size[5] - size[4]) + size[4]

    # boundary_coords = sample_boundary(N//10, epsilon, size=size, device=device)
    # coords = torch.cat((coords, boundary_coords), 0)
    return coords

def sample_boundary(N, epsilon=1e-3, size=(2, 2), device='cpu', boundary=['x0', 'x1', 'y0', 'y1']):
    bound_ranges_dict = {
        "x0": [[size[0], size[0] + epsilon], [size[2], size[3]], [size[4], size[5]]],
        "x1": [[size[1] - epsilon, size[1]], [size[2], size[3]], [size[4], size[5]]],
        "y0": [[size[0], size[1]], [size[2], size[2] + epsilon], [size[4], size[5]]],
        "y1": [[size[0], size[1]], [size[3] - epsilon, size[3]], [size[4], size[5]]],
        "z0": [[size[4], size[5]], [size[2], size[3]], [size[4], size[4] + epsilon]],
        "z1": [[size[4], size[5]], [size[2], size[3]], [size[5] - epsilon, size[5]]]
    }

    sides = boundary
    coords = []
    for side in sides:
        x_b, y_b, z_b = bound_ranges_dict[side]
        points = torch.empty(N, 3, device=device)
        points[..., 0] = torch.rand(N, device=device) * (x_b[1] - x_b[0]) + x_b[0]
        points[..., 1] = torch.rand(N, device=device) * (y_b[1] - y_b[0]) + y_b[0]
        points[..., 2] = torch.rand(N, device=device) * (z_b[1] - z_b[0]) + z_b[0]
        coords.append(points)
    coords = torch.cat(coords, dim=0)
    
    return coords