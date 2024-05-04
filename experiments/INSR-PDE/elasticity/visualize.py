from sklearn.metrics import zero_one_loss
import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def save_figure(fig, save_path, close=True):
    plt.savefig(save_path, bbox_inches='tight')
    if close:
        plt.close(fig)


def write_pointcloud_to_file(filename, points, color=None):
    pcd = o3d.geometry.PointCloud()
    if points.shape[1] == 3:
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        pcd.points = o3d.utility.Vector3dVector(np.hstack([points, np.zeros((points.shape[0], 1))]))
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(filename, pcd)


def draw_deformation_field3D(arr, vmin=None, vmax=None, color=None, plane_height=None, sphere_center=None, sphere_radius=None, hide_axis = False):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    if color is None:
        ax.scatter(arr[:,0], arr[:,1], arr[:,2], c = arr[:,0]+arr[:,1]+arr[:,2], cmap='cool', s=0.2)
    else:
        ax.scatter(arr[:,0], arr[:,1], arr[:,2], c = color, cmap='cool', s=0.2)
    ax.set_xlim3d(-3, 3)
    ax.set_ylim3d(-3, 3)
    ax.set_zlim3d(-3, 3)
    if hide_axis:
        ax.set_axis_off()
    # ax.view_init(0, 0)
    if plane_height is not None:
        X, Y = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
        Z = plane_height * np.ones_like(X)
        ax.plot_surface(X, Y, Z, alpha=0.2, color='green')  # the horizontal plane
    if sphere_radius is not None and sphere_center is not None:
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x = np.outer(np.cos(u), np.sin(v)) * sphere_radius
        y = np.outer(np.sin(u), np.sin(v)) * sphere_radius
        z = np.outer(np.ones(np.size(u)), np.cos(v)) * sphere_radius
        ax.plot_surface(x, y, z, linewidth=0.0, alpha=0.1, color='blue')

    return fig


def draw_deformation_field2D(arr, vmin=None, vmax=None, color=None, plane_height=None, circle_center=None, circle_radius=None, scatter_radius=0.2, hide_axis = False, cmap = 'cool'):
    fig, ax = plt.subplots(figsize=(3, 3))
    if color is None:
        ax.scatter(arr[:,0], arr[:,1], c = arr[:,0]+arr[:,1], cmap=cmap, s=scatter_radius)
    else:
        ax.scatter(arr[:,0], arr[:,1], c = color, cmap=cmap, s=scatter_radius)
    # plt.xlim(-3.5, 2.5)
    # plt.ylim(-3.5, 2.5)
    # plt.xlim(-2.5, 4.5)
    # plt.ylim(-3.5, 3.5)
    plt.xlim(-2.5, 2.5)
    plt.ylim(-3.5, 1.5)
    if plane_height is not None:
        plt.axhline(y=plane_height, color='green', linestyle='-', alpha=0.5)
    if circle_radius is not None and circle_center is not None:
        cir = plt.Circle(circle_center, circle_radius, color='orange',fill=False, linewidth=0.5)
        ax.set_aspect('equal', adjustable='datalim')
        ax.add_patch(cir)
    if hide_axis:
        ax.set_axis_off()
    else:
        plt.axhline(y=1.0, color='orange', linestyle='-', alpha=0.5)
        plt.axhline(y=-1.0, color='orange', linestyle='-', alpha=0.5)
    return fig


def draw_scalar_field2D(arr, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(3, 3))
    cax1 = ax.matshow(arr, vmin=vmin, vmax=vmax)
    fig.colorbar(cax1, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def draw_vector_field2D(u, v, tag=None):
    assert u.shape == v.shape
    indices = np.indices(u.shape)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.quiver(indices[0], indices[1], u, v, scale=u.shape[0], scale_units='width')
    if tag is not None:
        ax.text(-1, -1, tag, fontsize=12)
    return fig
     