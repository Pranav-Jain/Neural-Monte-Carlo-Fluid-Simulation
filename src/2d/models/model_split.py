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
        samples = self.sample_in_training()

        # dudt
        with torch.no_grad():
            prev_u = self.query_velocity(samples, use_prev=True).detach()

        if self.cfg.time_integration == 'semi_lag':
            # backtracking
            backtracked_position = samples - prev_u * dt # Eqn 9 in Rundi's paper INSR
            backtracked_position[..., 0] = torch.clamp(torch.clone(backtracked_position[..., 0]), min=self.scene_size[0], max=self.scene_size[1])
            backtracked_position[..., 1] = torch.clamp(torch.clone(backtracked_position[..., 1]), min=self.scene_size[2], max=self.scene_size[3])
            
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
        _, nearest_ind = kdtree.query(samples, k=10)
        for _ in range(1):
            mdn1 = np.median(grad_p[nearest_ind, 0], axis=1)
            mdn2 = np.median(grad_p[nearest_ind, 1], axis=1)
            grad_p[..., 0] = mdn1
            grad_p[..., 1] = mdn2
        
        return grad_p
    
    def get_area(self, v1, v2, v3):
        D = torch.ones((v1.shape[0], 3, 3))
        D[..., 0][..., 1:] = v1
        D[..., 1][..., 1:] = v2
        D[..., 2][..., 1:] = v3

        area = torch.abs(0.5 * torch.linalg.det(D))
        return area.to(self.device)

    
    def find_closest_index(self, s, grad):
        x_ind = torch.searchsorted(self.wost_samples_x, s[..., 0], side='right')
        y_ind = torch.searchsorted(self.wost_samples_y, s[..., 1], side='right')

        x2 = self.wost_samples_x[torch.clamp(x_ind, min=0, max=self.wost_samples_x.shape[0]-1)]
        y2 = self.wost_samples_y[torch.clamp(y_ind, min=0, max=self.wost_samples_y.shape[0]-1)]
        x1 = self.wost_samples_x[torch.clamp(x_ind-1, min=0, max=self.wost_samples_x.shape[0]-1)]
        y1 = self.wost_samples_y[torch.clamp(y_ind-1, min=0, max=self.wost_samples_y.shape[0]-1)]

        x = s[..., 0]
        y = s[..., 1]

        ind11 = torch.clamp((x_ind-1) * len(self.wost_samples_y) + (y_ind-1), min=0, max=grad.shape[0]-1)
        ind12 = torch.clamp((x_ind-1) * len(self.wost_samples_y) + (y_ind), min=0, max=grad.shape[0]-1)
        ind21 = torch.clamp((x_ind) * len(self.wost_samples_y) + (y_ind-1), min=0, max=grad.shape[0]-1)
        ind22 = torch.clamp((x_ind) * len(self.wost_samples_y) + (y_ind), min=0, max=grad.shape[0]-1) 

        w11 = (x2 - x)*(y2-y) / ((x2-x1)*(y2-y1))
        w12 = (x2 - x)*(y-y1) / ((x2-x1)*(y2-y1))
        w21 = (x - x1)*(y2-y) / ((x2-x1)*(y2-y1))
        w22 = (x - x1)*(y-y1) / ((x2-x1)*(y2-y1))

        # s_x = torch.linalg.norm(self.wost_samples[ind1][..., 0] - self.wost_samples[ind3][..., 0], axis=-1)
        # s_y = torch.linalg.norm(self.wost_samples[ind1][..., 1] - self.wost_samples[ind2][..., 1], axis=-1)
        
        # w1 = torch.linalg.norm(s[..., 0] - self.wost_samples[ind1][..., 0], axis=-1)/s_x
        # w2 = torch.linalg.norm(s[..., 1] - self.wost_samples[ind1][..., 1], axis=-1)/s_y

        # g = torch.clone(grad[ind1]).to(self.device)
        # g[..., 0] = (1-w1)*grad[ind1][..., 0] + w1*grad[ind3][..., 0]
        # g[..., 1] = (1-w2)*grad[ind1][..., 1] + w2*grad[ind2][..., 1]

        # return g
        out = w11[:, None]*grad[ind11] + w12[:, None]*grad[ind12] + w21[:, None]*grad[ind21] + w22[:, None]*grad[ind22]
        out[torch.isnan(out)] = 0.0
        return out
        # return (grad[ind1] + grad[ind2] + grad[ind3] + grad[ind4])/4.0

    # def find_closest_index(self, s, grad):
    #     grid = CubicBSplineGrid2d(resolution=grad.shape, n_channels=1)


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

        # grad_p = self.laplacian_smoothing(samples, grad_p)

        # grad_p[np.abs(grad_p) < 1e-3] = 0.0

        # min_x = np.min(grad_p[..., 0])
        # min_y = np.min(grad_p[..., 1])
        # max_x = np.max(grad_p[..., 0])
        # max_y = np.max(grad_p[..., 1])

        # grad_p[..., 0] = np.where((grad_p[..., 0] > max_x/2) & (grad_p[..., 0] > 0.0), 0.0, grad_p[..., 0])
        # grad_p[..., 0] = np.where((grad_p[..., 0] < min_x/2) & (grad_p[..., 0] < 0.0), 0.0, grad_p[..., 0])
        # grad_p[..., 1] = np.where((grad_p[..., 1] > max_y/2) & (grad_p[..., 1] > 0.0), 0.0, grad_p[..., 1])
        # grad_p[..., 1] = np.where((grad_p[..., 1] < min_y/2) & (grad_p[..., 1] < 0.0), 0.0, grad_p[..., 1])
        
        # self.wost_samples = torch.Tensor(samples).to(self.device)
        self.wost_samples_x = torch.Tensor(np.unique(samples[..., 0])).to(self.device)
        self.wost_samples_y = torch.Tensor(np.unique(samples[..., 1])).to(self.device)

        self.P = np.mean(p)

        # mask = (samples[..., 0] < self.scene_size[0] + self.bdry_eps) | (samples[..., 0] > self.scene_size[1] - self.bdry_eps) | (samples[..., 1] < self.scene_size[2] + self.bdry_eps) | (samples[..., 1] > self.scene_size[3] - self.bdry_eps)
        # grad_p[..., 0][mask] = 0.0
        # grad_p[..., 1][mask] = 0.0

        # print(self.wost_samples_x)
        # print(self.wost_samples_y)

        return samples, p, grad_p

    def get_divergence(self, resolution, save_path_png, save_path_pfm, vmin=None, vmax=None):
        grid_values, grid_samples = self.sample_velocity_field(resolution, to_numpy=False, return_samples=True, require_grad=True)
        div = divergence(grid_values, grid_samples).detach().cpu().numpy()
        div = -div[..., 0] # Wost solves lap u = -f

        min = np.min(div)
        max = np.max(div)
        print(div.shape)
        print(min)
        print(max)
        
        fig = draw_scalar_field2D(div, vmin=vmin, vmax=vmax, figsize=self.fig_size, cmap='viridis', colorbar=True)
        save_figure_nopadding(fig, save_path_png)
        return div

    @NeuralFluidBase._training_loop
    def _project_velocity(self, flag=False):
        """projection step for velocity: u <- u - grad(p)"""

        save_path_png = os.path.join(self.vis_mag_dir, f'mag_t{self.timestep:03d}.png')
        save_path_pfm = os.path.join(self.vis_mag_dir, f'mag_t{self.timestep:03d}.pfm')
        
        if not self.wost_flag:
            self.wost_flag=True
            self.pressure_samples = self.sample_in_training(resolution=self.cfg.wost_resolution)
            div = self.get_divergence(1000, save_path_png, save_path_pfm)
            
            samples_arr, p, grad_p = self.wost_pressure(div, save_path_pfm)

            # grad_p = self.laplacian_smoothing(samples_arr, grad_p)
            
            fig = self.draw_wost_pressure(p, samples_arr)
            save_path_pressure = os.path.join(self.vis_pressure_dir, f'p_t{self.timestep:03d}.png')
            save_figure_nopadding(fig, save_path_pressure)
            
            fig = self.draw_wost_pressure(grad_p[:, 0], samples_arr)
            save_path_pressure = os.path.join(self.vis_pressure_dir, f'gradp_x_t{self.timestep:03d}.png')
            save_figure_nopadding(fig, save_path_pressure)
            fig = self.draw_wost_pressure(grad_p[:, 1], samples_arr)
            save_path_pressure = os.path.join(self.vis_pressure_dir, f'gradp_y_t{self.timestep:03d}.png')
            save_figure_nopadding(fig, save_path_pressure)

            self.grad_p = torch.Tensor(grad_p).to(self.device)

        random_indices = torch.randint(0, self.pressure_samples.shape[0]-1, (int((self.cfg.sample_resolution))**2, ))
        samples = self.pressure_samples[random_indices]
        with torch.no_grad():
            prev_u = self.query_velocity(samples, use_prev=True).detach()

        target_u = prev_u - self.grad_p[random_indices]
        curr_u = self.query_velocity(samples)
        loss = torch.mean((curr_u - target_u) ** 2) # Eqn 11 in Rundi's paper INSR
        loss_dict = {'main': loss}

        return loss_dict

    def project_velocity(self):
        self._project_velocity()
    
    ################# visualization #####################

    def draw_wost_pressure(self, p_arr, samples):
        fig, ax = plt.subplots(figsize=self.fig_size)
        sc = ax.scatter(samples[:, 0], samples[:, 1], c=p_arr, cmap='viridis', s=0.1)
        ax.set_axis_off()
        plt.colorbar(sc)
        
        return fig