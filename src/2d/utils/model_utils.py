import torch

def sample_uniform_2D(resolution: int, normalize=True, with_boundary=False, size=None, device='cpu'):
    if (size[1] - size[0]) > (size[3] - size[2]):
        res_x, res_y = resolution, int(resolution * (size[3] - size[2]) / (size[1] - size[0]))
    else:
        res_x, res_y = int(resolution * (size[1] - size[0]) / (size[3] - size[2])), resolution

    x = torch.linspace(0.5, res_x - 0.5, res_x, device=device)
    y = torch.linspace(0.5, res_y - 0.5, res_y, device=device)
    if with_boundary:
        x = torch.cat([torch.tensor([0.0], device=device), x, torch.tensor([res_x * 1.0], device=device)])
        y = torch.cat([torch.tensor([0.0], device=device), y, torch.tensor([res_y * 1.0], device=device)])
    # coords = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
    coords = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1)
    if normalize:
        coords[..., 0] = coords[..., 0] / res_x * (size[1] - size[0]) + size[0]
        coords[..., 1] = coords[..., 1] / res_y * (size[3] - size[2]) + size[2]

    return coords

def sample_random_2D(N: int, normalize=True, size=None, device='cpu', epsilon=1e-1, obs_v=None):
    coords = torch.rand(N, 2, device=device)
    if normalize:
        coords[..., 0] = coords[..., 0] * (size[1] - size[0]) + size[0]
        coords[..., 1] = coords[..., 1] * (size[3] - size[2]) + size[2]

    # boundary_coords = sample_boundary(N//100, epsilon, size=size, device=device)
    # obstacle_coords = sample_obstacle(obs_v, epsilon, device=device)
    # coords = torch.cat((coords, boundary_coords), 0)
    return coords

def sample_boundary(N, epsilon=1e-3, size=(2, 2), device='cpu', boundary=['x0', 'x1', 'y0', 'y1']):
    bound_ranges_dict = {
        "x0": [[size[0], size[0] + epsilon], [size[2], size[3]]],
        "x1": [[size[1] - epsilon, size[1]], [size[2], size[3]]],
        "y0": [[size[0], size[1]], [size[2], size[2] + epsilon]],
        "y1": [[size[0], size[1]], [size[3] - epsilon, size[3]]],
    }

    sides = boundary
    coords = []
    for side in sides:
        x_b, y_b = bound_ranges_dict[side]
        points = torch.empty(N, 2, device=device)
        points[..., 0] = torch.rand(N, device=device) * (x_b[1] - x_b[0]) + x_b[0]
        points[..., 1] = torch.rand(N, device=device) * (y_b[1] - y_b[0]) + y_b[0]
        coords.append(points)
    coords = torch.cat(coords, dim=0)
    
    return coords

def sample_obstacle(obs_v, epsilon, device='cpu'):
    N = 50
    coords = []
    for v in obs_v:
        theta = torch.rand(N, device=device) * 2*torch.pi
        r = torch.rand(N, device=device) * epsilon
        points = torch.empty(N, 2, device=device)
        points[..., 0] = v[0] + r * torch.cos(theta)
        points[..., 1] = v[1] + r * torch.sin(theta)
        coords.append(points)
    coords = torch.cat(coords, dim=0)
    
    return coords

