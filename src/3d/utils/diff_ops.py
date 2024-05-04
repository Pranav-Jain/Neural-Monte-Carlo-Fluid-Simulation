# Based on https://github.com/vsitzmann/siren/blob/master/diff_operators.py
import torch
from torch.autograd import grad
import numpy as np


def hessian(y, x):
    """
    Hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, dim)
    return:
        shape (meta_batch_size, num_observations, dim, channels)
    """
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations,
                    y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y,
                                   create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status


def laplace(y, x, normalize=False, eps=0., return_grad=False):
    grad = gradient(y, x)
    if normalize:
        grad = grad / (grad.norm(dim=-1, keepdim=True) + eps)
    div = divergence(grad, x)

    if return_grad:
        return div, grad
    return div


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(
            y[..., i], x, torch.ones_like(y[..., i]),
            create_graph=True)[0][..., i:i+1]
    return div


def curl2d(y, x):
    curl = grad(
            y[..., 1], x, torch.ones_like(y[..., 1]),
            create_graph=True)[0][..., 0:0+1] \
            - \
            grad(
            y[..., 0], x, torch.ones_like(y[..., 0]),
            create_graph=True)[0][..., 1:1+1]
    return curl


def partial_derivative_x(field, element_length):
    diff = np.zeros_like(field)
    diff[1:-1, 1:-1] = (
        (
            field[2:  , 1:-1]
            -
            field[0:-2, 1:-1]
        ) / (
            2 * element_length
        )
    )
    return diff


def partial_derivative_y(field, element_length):
    diff = np.zeros_like(field)
    diff[1:-1, 1:-1] = (
        (
            field[1:-1, 2:  ]
            -
            field[1:-1, 0:-2]
        ) / (
            2 * element_length
        )
    )
    return diff


def curl2d_fdiff(vector_field, element_length):
    curl_applied = (
        partial_derivative_x(vector_field[..., 1], element_length)
        -
        partial_derivative_y(vector_field[..., 0], element_length)
    )
    return curl_applied


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def jacobian(y: torch.FloatTensor, x: torch.FloatTensor):

    """jacobian of y wrt x

    Args:
        y (torch.FloatTensor): (..., dim_y)
        x (torch.FloatTensor): (..., dim_x)

    Returns:
        jac (torch.FloatTensor): (..., dim_y, dim_x)
    """
    # N = y.shape[0]
    jac = torch.zeros(*y.shape[:-1], y.shape[-1], x.shape[-1]).to(y.device)

    for i in range(y.shape[-1]):
        y_i = y[..., i]
        jac[..., i, :] = grad(y_i, x, torch.ones_like(y_i), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status


# def jacobian(y, x):
#     """
#     Jacobian of y wrt x
#     y: shape (meta_batch_size, num_observations, channels)
#     x: shape (meta_batch_size, num_observations, dim)
#     ret: shape (meta_batch_size, num_observations, channels, dim)
#     """
#     meta_batch_size, num_observations = y.shape[:2]
#     # (meta_batch_size*num_points, 2, 2)
#     jac = torch.zeros(
#         meta_batch_size, num_observations,
#         y.shape[-1], x.shape[-1]).to(y.device)
#     for i in range(y.shape[-1]):
#         # calculate dydx over batches for each feature value of y
#         y_flat = y[...,i].view(-1, 1)
#         jac[:, :, i, :] = grad(
#             y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

#     status = 0
#     if torch.any(torch.isnan(jac)):
#         status = -1

#     return jac, status


if __name__ == '__main__':
    pass
