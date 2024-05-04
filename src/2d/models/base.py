import os
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import shutil
from tensorboardX import SummaryWriter
from .networks import get_network
from utils.diff_ops import jacobian, divergence, curl2d_fdiff, curl2d
from utils.model_utils import sample_uniform_2D, sample_random_2D
from utils.vis_utils import draw_scalar_field2D, draw_vector_field2D
import matplotlib.cm as cm
from PIL import Image
import os
from scipy.special import erf
import matplotlib.pyplot as plt


class NeuralFluidABC(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg.dt
        self.visc = cfg.visc
        self.diff = cfg.diff
        self.max_n_iters = cfg.max_n_iters
        self.sample_resolution = cfg.sample_resolution
        self.vis_resolution = cfg.vis_resolution
        self.timestep = 0
        self.boundary_cond = cfg.boundary_cond
        self.mode = cfg.mode
        self.tb = None
        self.sample_pattern = cfg.sample
        self.use_density = cfg.use_density

        self.scene_size = cfg.scene_size
        self.has_obstacle = False
        self.karman_vel = cfg.karman_vel
        self.center = None
        self.radius = None
        self.obstacle_vertices = None

        self.bdry_eps = cfg.bdry_eps
        
        self.fig_size = (int((self.scene_size[1] - self.scene_size[0]) * 10), int((self.scene_size[3] - self.scene_size[2]) * 10))

        # neural implicit network for velocity
        self.velocity_field = get_network(cfg, 2, 2).cuda()
        self.velocity_field_prev = get_network(self.cfg, 2, 2).cuda()
        self.velocity_field_tilde = get_network(self.cfg, 2, 2).cuda()
        self._set_require_grads(self.velocity_field_prev, False)
        self.device = torch.device("cuda:0")
        torch.cuda.empty_cache()

        self.P = 0.0
        
    @property
    def _trainable_networks(self):
        return {'velocity': self.velocity_field}

    def create_optimizer(self, use_scheduler=False, gamma=0.1, patience=1000, min_lr=1e-7, reset=False):
        self.loss_record = [10000, 0]
        # optimizer: use only one optimizer?
        param_list = []
        for net in self._trainable_networks.values():
            if reset:
                net.net.apply(net.weight_init)
                net.net[0].apply(net.first_layer_init)

            param_list.append({"params": net.parameters(), "lr": self.cfg.lr})
        self.optimizer = torch.optim.Adam(param_list)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_n_iters, 
        #     eta_min=min_lr, verbose=False) if use_scheduler else None
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, 
            min_lr=min_lr, patience=patience, verbose=False) if use_scheduler else None
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-7, max_lr=self.cfg.lr, 
        #     step_size_up=1000, step_size_down=1000, mode='triangular2', cycle_momentum=False) if use_scheduler else None

    @abstractmethod
    def step(self):
        raise NotImplementedError

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        if self.cfg.grad_clip > 0:
            param_list = []
            for net in self._trainable_networks.values():
                param_list = param_list + list(net.parameters())
            torch.nn.utils.clip_grad_norm_(param_list, 0.1)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step(loss_dict['main'])
            # self.scheduler.step()
    
    def _set_require_grads(self, model, require_grad):
        for p in model.parameters():
            p.requires_grad_(require_grad)
    
    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.cfg.model_dir, f"ckpt_step_t{self.timestep:03d}.pth")
        else:
            save_path = os.path.join(self.cfg.model_dir, f"ckpt_{name}.pth")

        save_dict = {}
        for name, net in self._trainable_networks.items():
            save_dict.update({f'net_{name}': net.cpu().state_dict()})
            net.cuda()
        save_dict.update({'timestep': self.timestep})

        torch.save(save_dict, save_path)
    
    def load_ckpt(self, name):
        """load saved checkpoint"""
        if type(name) is int:
            load_path = os.path.join(self.cfg.model_dir, f"ckpt_step_t{name:03d}.pth")
        else:
            load_path = os.path.join(self.cfg.model_dir, f"ckpt_{name}.pth")
        checkpoint = torch.load(load_path)

        for name, net in self._trainable_networks.items():
            net.load_state_dict(checkpoint[f'net_{name}'])
        self.timestep = checkpoint['timestep']

    @classmethod
    def _training_loop(cls, func):
        """a decorator function that warps a function inside a training loop"""
        tag = func.__name__
        def loop(self, *args, **kwargs):
            pbar = tqdm(range(self.max_n_iters))
            self.create_optimizer()
            
            for i in pbar:
                self.iter = i
                loss_dict = func(self, *args, **kwargs)
                self.update_network(loss_dict)

                loss_value = {k: v.item() for k, v in loss_dict.items()}

                pbar.set_description(f"{tag}[{self.timestep}]")
                pbar.set_postfix(loss_value)

                # if self.cfg.early_stop and ((tag == '_advect_velocity' and loss_value['main'] < 1e-6) or (tag == '_project_velocity' and loss_value['main'] < 1e-6)):
                if self.cfg.early_stop and loss_value['main'] <= 1.1e-10:
                    tqdm.write(f"early stopping at iteration {i}")
                    break

        return loop

    def add_obstacle(self, obs_sdf_func):
        self.has_obstacle = True
        self.obs_sdf_func = obs_sdf_func

    def query_velocity(self, samples, eps=1e-1, use_prev=False, use_tilde=False, use_bdry_cond=True):
        eps = self.bdry_eps

        if use_tilde:
            net_vel = self.velocity_field_tilde(samples)
        else:
            if use_prev:
                net_vel = self.velocity_field_prev(samples)
            else:
                net_vel = self.velocity_field(samples)

        if use_bdry_cond and self.cfg.src == 'karman':
            inlet_mask = (samples[..., 0]>=self.scene_size[0]) & (samples[..., 0]<=self.scene_size[0]+eps)
            net_vel[..., 0][inlet_mask] = self.cfg.karman_vel

            net_vel = self.smoothstep_circular_obs(samples, net_vel, eps=eps)

            u_weight = torch.ones_like(samples[..., 0])
            v_weight = torch.min(torch.abs(samples[..., 1] - (self.scene_size[2])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 1] - (self.scene_size[3])).clamp(min=0, max=eps)) / eps
            weight = torch.stack([u_weight, v_weight], dim=-1).detach()

            net_vel = weight * net_vel

        elif use_bdry_cond and self.cfg.src == 'taylorgreen':
            u_weight = torch.min(torch.abs(samples[..., 0] - (self.scene_size[0])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 0] - (self.scene_size[1])).clamp(min=0, max=eps)) / eps
            v_weight = torch.min(torch.abs(samples[..., 1] - (self.scene_size[2])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 1] - (self.scene_size[3])).clamp(min=0, max=eps)) / eps
            weight = torch.stack([u_weight, v_weight], dim=-1).detach()

            net_vel = weight * net_vel

        elif use_bdry_cond and self.cfg.src == 'jpipe':
            inlet_mask = (samples[..., 0]>=0.0) & (samples[..., 0]<=0.1) & (samples[..., 1]>=0.0) & (samples[..., 1]<=0.5)
            net_vel[..., 0][inlet_mask] = self.cfg.karman_vel

            mask1 = (samples[..., 0] >= 0.0) & (samples[..., 0] <= 1.0)
            mask2 = (samples[..., 1] >= 1.0) & (samples[..., 1] <= 2.0)
            mask = ~mask1 & ~mask2

            n = samples[mask] - torch.Tensor([1.,1.]).cuda()
            n = n/torch.linalg.norm(n, axis=0)
            u_n = ((n[...,0]*net_vel[mask][...,0])+(n[...,1]*net_vel[mask][...,1]))[:,None]*n
            ut = net_vel[mask] - u_n
            dist = self.obs_sdf_func(samples[mask])
            net_vel[mask] = ut + dist[:,None]*u_n
            
            u_weight = torch.ones_like(net_vel[..., 0])
            v_weight = torch.ones_like(net_vel[..., 1])

            v_weight[mask1] = torch.min(torch.abs(samples[..., 1][mask1] - 0.5).clamp(min=0, max=eps),
                                torch.abs(samples[..., 1][mask1]).clamp(min=0, max=eps)) / eps
            u_weight[mask2] = torch.min(torch.abs(samples[..., 0][mask2] - 1.5).clamp(min=0, max=eps),
                                torch.abs(samples[..., 0][mask2] - 2.0).clamp(min=0, max=eps)) / eps
            
            weight = torch.stack([u_weight, v_weight], dim=-1).detach()
            net_vel = weight * net_vel

            d = torch.sqrt((samples[..., 0]-1)**2 + (samples[..., 1]-1)**2)
            mask1 = (samples[..., 0]>=0.0) & (samples[..., 0]<=1.0) & (samples[..., 1]>=0.0) & (samples[..., 1]<=0.5)
            mask2 = (samples[..., 0]>=1.5) & (samples[..., 0]<=2.0) & (samples[..., 1]>=1.0) & (samples[..., 1]<=2.0)
            mask3 = (d >= 0.5) & (d <= 1.0) & (samples[..., 0]>=1.0) & (samples[..., 1]<=1.0)
            mask = mask1 | mask2 | mask3
            net_vel[~mask] = 0.0

        return net_vel

    def sample_in_training(self, resolution=None):
        if resolution == None:
            resolution = self.sample_resolution
        if self.sample_pattern == 'random':
            samples = sample_random_2D(resolution**2, size=self.scene_size, device=self.device, epsilon=self.bdry_eps, obs_v=self.obstacle_vertices).requires_grad_(True)
        elif self.sample_pattern == 'uniform':
            samples = sample_uniform_2D(resolution, with_boundary=True, size=self.scene_size, device=self.device).requires_grad_(True)
        elif self.sample_pattern == 'random+uniform':
            samples = torch.cat([sample_random_2D(resolution**2//2, size=self.scene_size, device=self.device),
                        sample_uniform_2D(resolution//2, with_boundary=True, size=self.scene_size, device=self.device).view(-1, 2)], dim=0).requires_grad_(True)
        else:
            raise NotImplementedError
        
        if self.cfg.src == 'karman':
            dist = self.obs_sdf_func(samples)
            samples = samples[dist > 0]

        if self.cfg.src == 'jpipe':
            d = torch.sqrt((samples[..., 0]-1)**2 + (samples[..., 1]-1)**2)
            mask1 = (samples[..., 0]>=0.0) & (samples[..., 0]<=1.0) & (samples[..., 1]>=0.0) & (samples[..., 1]<=0.5)
            mask2 = (samples[..., 0]>=1.5) & (samples[..., 0]<=2.0) & (samples[..., 1]>=1.0) & (samples[..., 1]<=2.0)
            mask3 = (d >= 0.5) & (d <= 1.0) & (samples[..., 0]>=1.0) & (samples[..., 1]<=1.0)
            mask = mask1 | mask2 | mask3
            samples = samples[mask]

        return samples

    def sample_velocity_field(self, resolution, to_numpy=True, with_boundary=True, return_samples=False, require_grad=False):
        grid_samples = sample_uniform_2D(resolution, with_boundary=with_boundary, size=self.scene_size, device=self.device)
        if require_grad:
            grid_samples = grid_samples.requires_grad_(True)

        out = self.query_velocity(grid_samples, use_prev=True, use_bdry_cond=True)

        if to_numpy:
            out = out.detach().cpu().numpy()
            grid_samples = grid_samples.detach().cpu().numpy()
        if return_samples:
            return out, grid_samples
        return out

    def draw(self, tag, resolution, **kwargs):
        func_str = f'draw_{tag}'
        try:
            return getattr(self, func_str)(resolution, **kwargs)
        except Exception as e:
            print(e)
            print(f"no method named '{func_str}'.")
            exit()

    def draw_velocity(self, resolution, save_txt_v = None, save_txt_s = None):
        grid_values, grid_samples = self.sample_velocity_field(resolution, to_numpy=True, with_boundary=True, return_samples=True)

        if save_txt_v is not None:
            np.savetxt(save_txt_v, grid_values.reshape((grid_values.shape[0]*grid_values.shape[1], grid_values.shape[2])))
            np.savetxt(save_txt_s, grid_samples.reshape((grid_samples.shape[0]*grid_samples.shape[1], grid_samples.shape[2])))
        x, y = grid_samples[..., 0], grid_samples[..., 1]
        fig = draw_vector_field2D(grid_values[..., 0], grid_values[..., 1], x, y, c=self.center, r=self.radius, figsize=self.fig_size)
        return fig

    def draw_vorticity(self, resolution, save_txt_v = None, save_txt_s = None, vmin=-5, vmax=5):
        grid_values, grid_samples = self.sample_velocity_field(resolution, to_numpy=False, return_samples=True, require_grad=True)
        curl = curl2d(grid_values, grid_samples).squeeze(-1).detach().cpu().numpy()
        grid_samples = grid_samples.detach().cpu().numpy()

        if save_txt_v is not None:
            np.savetxt(save_txt_v, curl.reshape((curl.shape[0]*curl.shape[1], 1)))
            np.savetxt(save_txt_s, grid_samples.reshape((grid_samples.shape[0]*grid_samples.shape[1], grid_samples.shape[2])))

        fig = draw_scalar_field2D(curl, vmin=vmin, vmax=vmax, figsize=self.fig_size, cmap='bwr', colorbar=False)
        return fig

    def draw_density(self, resolution, vmin=0, vmax=1):
        p = self.sample_density_field(resolution, to_numpy=True)[..., 0]
        fig = draw_scalar_field2D(p, vmin, vmax, figsize=self.fig_size)
        return fig

    def compute_kinetic_energy(self, resolution):
        grid_values = self.sample_velocity_field(resolution, to_numpy=True, with_boundary=False)
        Ek = 0.5 * np.mean(grid_values ** 2) + self.P
        return Ek


class NeuralFluidBase(NeuralFluidABC):
    def __init__(self, cfg):
        super(NeuralFluidBase, self).__init__(cfg)

    @NeuralFluidABC._training_loop
    def _add_source(self, source_func, is_init=True):
        """forward computation for add source"""
        
        samples = self.sample_in_training()

        out_rand = self.query_velocity(samples, use_bdry_cond=True)
        if is_init:
            target_rand_val = source_func(samples)
        else:
            raise NotImplemented

        loss_random = F.mse_loss(out_rand, target_rand_val)
        loss_dict = {'main': loss_random}

        return loss_dict
    
    def add_source(self, attr: str, source_func, is_init=True):
        self.create_optimizer()
        self.source_func = source_func
        self._add_source(source_func, is_init)
        self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())
        self.save_ckpt()

    def smoothstep(self, x, xmin, xmax, eps=1e-1, upper=True):
        lower_mask = (x >= xmin) & (x <= eps + xmin)
        middle_mask = (x > eps + xmin) & (x < xmax - eps)
        upper_mask = (x >= xmax - eps) & (x <= xmax)

        l = torch.ones_like(x, device=self.device)
        l[lower_mask] = self._smoothstep_linear(x[lower_mask], xmin, eps)
        l[middle_mask] = 1.0
        if upper:
            l[upper_mask] = self._smoothstep_linear(x[upper_mask], xmax, eps)
        else:
            l[upper_mask] = 1.0
        
        return l
    
    def smoothstep_circular_obs(self, samples, vel, eps=1e-1): # Implements no slip
        dist = self.obs_sdf_func(samples)
        threshold = eps
        weight = torch.clamp(dist, 0, threshold) / threshold
        vel *= weight.unsqueeze(-1)

        return vel

    def _smoothstep_poly(self, x, xm, e):
        y = torch.abs(x-xm)
        return y + (((3-2*e)/e**2) * y**2) + (((e-2)/e**3) * y**3)
    
    def _smoothstep_poly2(self, x, xm, e):
        y = torch.abs(x-xm)
        return y + (((2-2*e)/e**2) * y**2) + (((e-1)/e**3) * y**3)
        
    def _smoothstep_tanh(self, x, xm, e):
        y = torch.abs(x-xm)
        return ((torch.exp(y) - torch.exp(-y))/(torch.exp(y) + torch.exp(-y))) * ((np.exp(e) + np.exp(-e))/(np.exp(e) - np.exp(-e)))
        
    def _smoothstep_logsigmoid(self, x, xm, e):
        y = torch.abs(x-xm)
        return torch.log(2/(1+torch.exp(-y))) / np.log(2/(1+np.exp(-e)))
        
    def _smoothstep_sigmoid(self, x, xm, e):
        y = torch.abs(x-xm)
        return ((1 - torch.exp(-y))/(1 + torch.exp(-y))) * ((1 + np.exp(-e))/(1 - np.exp(-e)))
    
    def _smoothstep_linear(self, x, xm, e):
        y = torch.abs(x-xm)
        return y/e
    
    def _smoothstep_sin(self, x, xm, e):
        y = torch.abs(x - xm)
        return torch.sin(y/e * np.pi/2)