import os
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from .base import NeuralFluidBase
from .networks import get_network
from utils.diff_ops import curl2d_fdiff, laplace, divergence, jacobian, gradient, curl2d
from utils.model_utils import sample_uniform_2D, sample_random_2D
from utils.vis_utils import draw_scalar_field2D, draw_vector_field2D, save_figure, save_figure_nopadding
import zombie_bindings
import json
import matplotlib.pyplot as plt
import sys
import cv2
from sklearn.neighbors import KDTree
import gpytoolbox
from torch_cubic_spline_grids import CubicBSplineGrid2d

np.set_printoptions(threshold=sys.maxsize)


class NeuralFluidSplit(NeuralFluidBase):
    def __init__(self, cfg):
        super(NeuralFluidSplit, self).__init__(cfg)

        f = open(self.cfg.wost_json)
        self.wost_data = json.load(f)
        f.close()

        self.vis_mag_dir = os.path.join(self.cfg.results_dir, 'magnitude')
        self.vis_pressure_dir = os.path.join(self.cfg.results_dir, 'pressure')
        self.grad_p = None
        self.pressure_samples = None
        self.wost_samples_x = None
        self.wost_samples_y = None
        self.wost_samples_z = None
        self.wost_samples = None
        self.wost_flag = False

    @property
    def _trainable_networks(self):
        return {'velocity': self.velocity_field}

    def step(self):
        if self.cfg.reset_wts == 1:
            reset = True
        else:
            reset=False

        self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())
        self.velocity_field_tilde.load_state_dict(self.velocity_field.state_dict())

        if self.cfg.adv_ref == 0:
            self.create_optimizer(reset=reset)    
            self.advect_velocity(dt=self.cfg.dt, flag=False)

            self.velocity_field_tilde.load_state_dict(self.velocity_field.state_dict()) #u_tilde
            self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())
            self.create_optimizer(reset=reset)
            self.project_velocity()
            self.wost_flag = False

        else:
            self.create_optimizer(reset=reset)
            self.advect_velocity(dt=self.cfg.dt/2, flag=False)

            self.velocity_field_tilde.load_state_dict(self.velocity_field.state_dict()) #u_tilde
            self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())
            self.create_optimizer(reset=reset)
            self.project_velocity()
            self.wost_flag = False

            self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())
            self.create_optimizer(reset=reset)
            self.advect_velocity(dt=self.cfg.dt/2, flag=True)

            self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict()) #u_tilde
            self.create_optimizer(reset=reset)
            self.project_velocity()
            self.wost_flag = False

        self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())

    def advect_velocity(self, dt, flag):
        self._advect_velocity(dt, flag)
    
    @NeuralFluidBase._training_loop
    def _advect_velocity(self, dt, flag):
        """velocity advection: dudt = -(u\cdot grad)u"""
        samples = self.sample_in_training(apply_obstable=False)

        # dudt
        with torch.no_grad():
            prev_u = self.query_velocity(samples, use_prev=True).detach()

        if self.cfg.time_integration == 'semi_lag':
            # backtracking
            backtracked_position = samples - prev_u * dt
            backtracked_position[:, 0] = torch.clamp(torch.clone(backtracked_position[:, 0]), min=self.scene_size[0], max=self.scene_size[1])
            backtracked_position[:, 1] = torch.clamp(torch.clone(backtracked_position[:, 1]), min=self.scene_size[2], max=self.scene_size[3])
            backtracked_position[:, 2] = torch.clamp(torch.clone(backtracked_position[:, 2]), min=self.scene_size[4], max=self.scene_size[5])
            
            with torch.no_grad():
                if not flag:
                    advected_u = self.query_velocity(backtracked_position, use_prev=True).detach()
                else:
                    advected_u = 2*self.query_velocity(backtracked_position, use_prev=True).detach() - self.query_velocity(backtracked_position, use_tilde=True)

            curr_u = self.query_velocity(samples)
            
            loss = torch.mean((curr_u - advected_u) ** 2)
            loss_dict = {'main': loss}

        else:
            raise NotImplementedError

        # if self.boundary_cond == 'zero':
        #     bc_loss_dict = self._velocity_boundary_loss(samples.shape[0] // 100)
        #     loss_dict.update(bc_loss_dict)

        return loss_dict

    def laplacian_smoothing(self, samples, grad_p, lda=1.0):
        kdtree = KDTree(samples)
        _, nearest_ind = kdtree.query(samples, k=8)
        for _ in range(1):
            mdn1 = np.median(grad_p[nearest_ind, 0], axis=1)
            mdn2 = np.median(grad_p[nearest_ind, 1], axis=1)
            mdn3 = np.median(grad_p[nearest_ind, 2], axis=1)
            grad_p[..., 0] = mdn1
            grad_p[..., 1] = mdn2
            grad_p[..., 2] = mdn3

            # mn1 = np.mean(grad_p[nearest_ind, 0], axis=1)
            # mn2 = np.mean(grad_p[nearest_ind, 1], axis=1)
            # grad_p[..., 0] = mn1
            # grad_p[..., 1] = mn2
        
        return grad_p
    
    def find_closest_index(self, s, grad):
        x_ind = torch.searchsorted(self.wost_samples_x, s[..., 0], side='right')
        y_ind = torch.searchsorted(self.wost_samples_y, s[..., 1], side='right')
        z_ind = torch.searchsorted(self.wost_samples_z, s[..., 2], side='right')

        x1 = self.wost_samples_x[torch.clamp(x_ind, min=0, max=self.wost_samples_x.shape[0]-1)]
        y1 = self.wost_samples_y[torch.clamp(y_ind, min=0, max=self.wost_samples_y.shape[0]-1)]
        z1 = self.wost_samples_z[torch.clamp(z_ind, min=0, max=self.wost_samples_z.shape[0]-1)]
        x0 = self.wost_samples_x[torch.clamp(x_ind-1, min=0, max=self.wost_samples_x.shape[0]-1)]
        y0 = self.wost_samples_y[torch.clamp(y_ind-1, min=0, max=self.wost_samples_y.shape[0]-1)]
        z0 = self.wost_samples_z[torch.clamp(z_ind-1, min=0, max=self.wost_samples_z.shape[0]-1)]

        x = s[..., 0]
        y = s[..., 1]
        z = s[..., 2]

        xd = (x-x0)/(x1-x0)
        yd = (y-y0)/(y1-y0)
        zd = (z-z0)/(z1-z0)
        xd = xd[:, None, None]
        yd = yd[:, None, None]
        zd = zd[:, None, None]

        ind111 = torch.clamp(((x_ind) *len(self.wost_samples_y)*len(self.wost_samples_z)) + ((y_ind)*len(self.wost_samples_z)) + z_ind, min=0, max=grad.shape[0]-1)
        ind110 = torch.clamp(((x_ind) *len(self.wost_samples_y)*len(self.wost_samples_z)) + ((y_ind)*len(self.wost_samples_z)) + z_ind-1, min=0, max=grad.shape[0]-1)
        ind101 = torch.clamp(((x_ind) *len(self.wost_samples_y)*len(self.wost_samples_z)) + ((y_ind-1)*len(self.wost_samples_z)) + z_ind, min=0, max=grad.shape[0]-1)
        ind011 = torch.clamp(((x_ind-1) *len(self.wost_samples_y)*len(self.wost_samples_z)) + ((y_ind)*len(self.wost_samples_z)) + z_ind, min=0, max=grad.shape[0]-1)
        ind001 = torch.clamp(((x_ind-1) *len(self.wost_samples_y)*len(self.wost_samples_z)) + ((y_ind-1)*len(self.wost_samples_z)) + z_ind, min=0, max=grad.shape[0]-1)
        ind010 = torch.clamp(((x_ind-1) *len(self.wost_samples_y)*len(self.wost_samples_z)) + ((y_ind)*len(self.wost_samples_z)) + z_ind-1, min=0, max=grad.shape[0]-1)
        ind100 = torch.clamp(((x_ind) *len(self.wost_samples_y)*len(self.wost_samples_z)) + ((y_ind-1)*len(self.wost_samples_z)) + z_ind-1, min=0, max=grad.shape[0]-1)
        ind000 = torch.clamp(((x_ind-1) *len(self.wost_samples_y)*len(self.wost_samples_z)) + ((y_ind-1)*len(self.wost_samples_z)) + z_ind-1, min=0, max=grad.shape[0]-1)

        c00 = grad[ind000]*(1-xd) + grad[ind100]*xd
        c01 = grad[ind001]*(1-xd) + grad[ind101]*xd
        c10 = grad[ind010]*(1-xd) + grad[ind110]*xd
        c11 = grad[ind011]*(1-xd) + grad[ind111]*xd

        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd

        out = c0*(1-zd) + c1*zd

        out[torch.isnan(out)] = 0.0
        out[torch.abs(out) == torch.inf] = 0.0
        # print(out)
        return out


    def wost_pressure(self, div, mag_path):
        sceneConfig = self.wost_data["scene"]
        sceneConfig["sourceValue"] = mag_path
        solverConfig = self.wost_data["solver"]
        outputConfig = self.wost_data["output"]

        scene = zombie_bindings.Scene(sceneConfig, div)
        # scene = zombie_bindings.Scene(sceneConfig)
        
        samples, p_arr, grad_arr = zombie_bindings.wost(scene, solverConfig, outputConfig, self.pressure_samples.detach().cpu().numpy())
        samples = np.array(samples)
        p = np.array(p_arr)
        grad_p = np.array(grad_arr)

        # grad_p[np.isnan(grad_p)] = 0.0
        
        # min_x = np.min(grad_p[..., 0])
        # min_y = np.min(grad_p[..., 1])
        # max_x = np.max(grad_p[..., 0])
        # max_y = np.max(grad_p[..., 1])

        # grad_p[..., 0] = np.where((grad_p[..., 0] > max_x/2) & (grad_p[..., 0] > 0.0), 0.0, grad_p[..., 0])
        # grad_p[..., 0] = np.where((grad_p[..., 0] < min_x/2) & (grad_p[..., 0] < 0.0), 0.0, grad_p[..., 0])
        # grad_p[..., 1] = np.where((grad_p[..., 1] > max_y/2) & (grad_p[..., 1] > 0.0), 0.0, grad_p[..., 1])
        # grad_p[..., 1] = np.where((grad_p[..., 1] < min_y/2) & (grad_p[..., 1] < 0.0), 0.0, grad_p[..., 1])
        
        # self.wost_samples = torch.Tensor(samples).to(self.device)
        # self.wost_samples_x = torch.Tensor(np.unique(samples[..., 0])).to(self.device)
        # self.wost_samples_y = torch.Tensor(np.unique(samples[..., 1])).to(self.device)
        # self.wost_samples_z = torch.Tensor(np.unique(samples[..., 2])).to(self.device)

        self.P = np.mean(p)

        # mask = (samples[..., 0] < self.scene_size[0] + self.bdry_eps) | (samples[..., 0] > self.scene_size[1] - self.bdry_eps) | (samples[..., 1] < self.scene_size[2] + self.bdry_eps) | (samples[..., 1] > self.scene_size[3] - self.bdry_eps)
        # grad_p[..., 0][mask] = 0.0
        # grad_p[..., 1][mask] = 0.0

        # print(self.wost_samples_x)
        # print(self.wost_samples_y)

        return samples, p, grad_p

    def get_divergence(self, resolution, save_path_png, save_path_pfm, vmin=None, vmax=None):
        grid_values, grid_samples = self.sample_velocity_field(resolution, to_numpy=False, return_samples=True, require_grad=True, flatten=False)
        div = divergence(grid_values, grid_samples).detach().cpu().numpy()
        div = -div[..., 0]

        min = np.min(div)
        max = np.max(div)
        print(div.shape)
        print(min)
        print(max)

        self.draw_scalar(grid_samples.reshape((-1, 3)).detach().cpu().numpy(), div.reshape((-1, 1)), save_path_png)
        
        ######################## Visualization TODO ##########################################
        # fig = draw_scalar_field2D(grid_samples, div, vmin=vmin, vmax=vmax, figsize=self.fig_size, cmap='viridis', colorbar=True)
        # save_figure_nopadding(fig, save_path_png)
        return div

    @NeuralFluidBase._training_loop
    def _project_velocity(self, flag=False):
        """projection step for velocity: u <- u - grad(p)"""
        
        # samples = self.sample_in_training(resolution=self.cfg.sample_resolution)
        # if self.pressure_samples is None:
        #     samples = self.sample_in_training(resolution=self.cfg.wost_resolution)
        # else:
        #     samples = torch.Tensor(self.pressure_samples).to(self.device)
        # samples_arr = samples.detach().cpu().numpy()
        # samples = self.sample_in_training(resolution=self.cfg.wost_resolution)

        save_path_png = os.path.join(self.vis_mag_dir, f'mag_t{self.timestep:03d}.png')
        save_path_pfm = os.path.join(self.vis_mag_dir, f'mag_t{self.timestep:03d}.pfm')
        
        if not self.wost_flag:
            self.wost_flag=True
            self.pressure_samples = self.sample_in_training(resolution=self.cfg.wost_resolution)
            div = self.get_divergence(self.cfg.vis_resolution, save_path_png, save_path_pfm)
            
            samples_arr, p, grad_p = self.wost_pressure(div, save_path_pfm)

            # grad_p[:, [1, 0, 2]] = grad_p[:, [0, 1, 2]]
            # self.pressure_samples = torch.Tensosamples_arr
            # grad_p = self.laplacian_smoothing(samples_arr, grad_p)
            
            ######################## Visualization TODO ##########################################
            # fig = self.draw_wost_pressure(p, samples_arr)
            save_path_pressure = os.path.join(self.vis_pressure_dir, f'p_t{self.timestep:03d}.png')
            self.draw_scalar(samples_arr, p, save_path_pressure)
            # save_figure_nopadding(fig, save_path_pressure)
            
            # fig = self.draw_wost_pressure(grad_p[:, 0], samples_arr)
            # save_path_pressure = os.path.join(self.vis_pressure_dir, f'gradp_x_t{self.timestep:03d}.png')
            # save_figure_nopadding(fig, save_path_pressure)
            # fig = self.draw_wost_pressure(grad_p[:, 1], samples_arr)
            # save_path_pressure = os.path.join(self.vis_pressure_dir, f'gradp_y_t{self.timestep:03d}.png')
            # save_figure_nopadding(fig, save_path_pressure)

            self.grad_p = torch.Tensor(grad_p).to(self.device)
            # self.grad_p = grad_p
        else:
            # samples = torch.clone(self.pressure_samples)
            plt.close("all")

        random_indices = torch.randint(0, self.pressure_samples.shape[0], (int((self.cfg.sample_resolution))**2, ))
        samples = self.pressure_samples[random_indices]
        with torch.no_grad():
            prev_u = self.query_velocity(samples, use_prev=True).detach()

        # grad_p = self.get_gradient(self.pressure_samples, self.grad_p)
        # grad_p = scipy.interpolate.griddata(self.pressure_samples, self.grad_p, samples.detach().cpu().numpy(), method='nearest')
        # grad_p = self.find_closest_index(samples, self.grad_p)

        # fig, ax = plt.subplots(figsize=self.fig_size)
        # sc = ax.scatter(samples.detach().cpu().numpy()[:, 0], samples.detach().cpu().numpy()[:, 1], c=grad_p[..., 0].detach().cpu().numpy(), cmap='viridis', s=0.1)
        # # ax.set_axis_off()
        # plt.colorbar(sc)
        # plt.savefig("out.png")
        # exit()

        target_u = prev_u - self.grad_p[random_indices]
        curr_u = self.query_velocity(samples)
        loss = torch.mean((curr_u - target_u) ** 2)
        loss_dict = {'main': loss}

        return loss_dict

    def project_velocity(self):
        self._project_velocity()
    
    # def sample_pressure_field(self, resolution, to_numpy=True, with_boundary=False, return_samples=False, require_grad=False):
    #     grid_samples = sample_uniform_2D(resolution, with_boundary=with_boundary, size=self.scene_size, device=self.device)
    #     if require_grad:
    #         grid_samples = grid_samples.requires_grad_(True)

    #     out = self.pressure_field(grid_samples)
    #     if to_numpy:
    #         out = out.detach().cpu().numpy()
    #         grid_samples = grid_samples.detach().cpu().numpy()
    #     if return_samples:
    #         return out, grid_samples
    #     return out

    # def draw_pressure(self, resolution):
    #     p = self.sample_pressure_field(resolution, to_numpy=True)
    #     fig = draw_scalar_field2D(p, figsize=self.fig_size)
    #     return fig