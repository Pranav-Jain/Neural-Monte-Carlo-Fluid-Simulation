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
import pyopenvdb as vdb

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
        
        self.fig_size = (int((self.scene_size[1] - self.scene_size[0]) * 5), int((self.scene_size[3] - self.scene_size[2]) * 5))
        # self.fig_size = (self.scene_size[0], self.scene_size[1])

        # neural implicit network velocity field
        self.velocity_field = get_network(cfg, 3, 3).cuda()
        self.velocity_field_prev = get_network(self.cfg, 3, 3).cuda()
        self.velocity_field_tilde = get_network(self.cfg, 3, 3).cuda()
        self._set_require_grads(self.velocity_field_prev, False)
        self._set_require_grads(self.velocity_field_tilde, False)
        self.device = torch.device("cuda:0")

        self.P = 0.0
        
    @property
    def _trainable_networks(self):
        return {'velocity': self.velocity_field}

    def create_optimizer(self, use_scheduler=False, gamma=0.1, patience=500, min_lr=1e-8, reset=False):
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
            min_lr=min_lr, patience=patience, verbose=True) if use_scheduler else None
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
        # self.update_loss_record(loss_dict['main'].item())
    
    def update_loss_record(self, loss_val):
        if loss_val < self.loss_record[0]:
            self.loss_record = [loss_val, 0]
        else:
            self.loss_record[1] += 1
    
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
            # if tag == '_add_source':
            #     pbar = tqdm(range(5000))
            self.create_optimizer()
            # self.tb.train_iter = 0
            for i in pbar:
                self.iter = i
                loss_dict = func(self, *args, **kwargs)
                self.update_network(loss_dict)

                loss_value = {k: v.item() for k, v in loss_dict.items()}

                # self.tb.add_scalars(tag, loss_value, global_step=i)
                # self.tb.train_iter += 1
                pbar.set_description(f"{tag}[{self.timestep}]")
                pbar.set_postfix(loss_value)

                # if self.cfg.early_stop and ((self.iter > 500 and tag == '_advect_velocity' and loss_value['main'] < 1e-7) or (self.iter > 500 and tag == '_project_velocity' and loss_value['main'] < 1e-6)):
                if self.cfg.early_stop and loss_value['main'] <= 1.1e-10:
                    tqdm.write(f"early stopping at iteration {i}")
                    break
                # elif self.cfg.early_stop and self.optimizer.param_groups[0]['lr'] <= 1.1e-8:
                #     self.optimizer.param_groups[0]['lr'] = self.cfg.lr
                    
        return loop

    def add_obstacle(self, obs_sdf_func):
        self.has_obstacle = True
        self.obs_sdf_func = obs_sdf_func

    def query_velocity(self, samples, eps=1e-1, use_prev=False, use_tilde=False, use_bdry_cond=True):
        # FIXME: this is for karman vortex
        eps = self.bdry_eps
        # features = torch.stack([samples[..., 0], 
        #                         samples[..., 1],
        #                         samples[..., 2], 
        #                         torch.sin(samples[..., 0]), 
        #                         torch.sin(samples[..., 1]),
        #                         torch.sin(samples[..., 2]), 
        #                         torch.sin(2*samples[..., 0]), 
        #                         torch.sin(2*samples[..., 1]),
        #                         torch.sin(2*samples[..., 2]),
        #                         torch.cos(samples[..., 0]),
        #                         torch.cos(samples[..., 1]),
        #                         torch.cos(samples[..., 2]),
        #                         torch.cos(2*samples[..., 0]),
        #                         torch.cos(2*samples[..., 1]),
        #                         torch.cos(2*samples[..., 2])], dim=-1)
        if use_tilde:
            net_vel = self.velocity_field_tilde(samples)
        else:
            if use_prev:
                net_vel = self.velocity_field_prev(samples)
            else:
                net_vel = self.velocity_field(samples)

        
        if use_bdry_cond and self.cfg.src == 'smoke':
            center1 = torch.Tensor([0.0, 0.0, -0.6]).cuda()
            radius1 = 0.1
            dist = torch.linalg.norm(samples - center1, axis=-1)
            mask1 = dist < radius1

            np.random.seed(self.timestep)
            r = torch.Tensor(np.random.random(net_vel[mask1].shape[0])).cuda()
            r = r*2 - 1
            r *= 10
            net_vel[..., 0][mask1] = 0.01*r
            net_vel[..., 1][mask1] = 0.01*r
            net_vel[..., 2][mask1] = 0.2+0.01*r
            

            u_weight = torch.min(torch.abs(samples[..., 0] - (self.scene_size[0])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 0] - (self.scene_size[1])).clamp(min=0, max=eps),) / eps
            v_weight = torch.min(torch.abs(samples[..., 1] - (self.scene_size[2])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 1] - (self.scene_size[3])).clamp(min=0, max=eps)) / eps
            w_weight = torch.min(torch.abs(samples[..., 2] - (self.scene_size[4])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 2] - (self.scene_size[5])).clamp(min=0, max=eps)) / eps
            weight = torch.stack([u_weight, v_weight, w_weight], dim=-1).detach()

            net_vel = weight * net_vel

        elif use_bdry_cond and self.cfg.src == 'smoke_obs':
            center1 = torch.Tensor([0.0, 0.0, -0.6]).cuda()
            radius1 = 0.1
            dist = torch.linalg.norm(samples - center1, axis=-1)
            mask1 = dist < radius1
            net_vel[..., 2][mask1] = 1.0
            
            dist = self.obs_sdf_func(samples)
            threshold = eps
            weight = torch.clamp(dist, 0, threshold) / threshold
            net_vel *= weight.unsqueeze(-1)

            u_weight = torch.min(torch.abs(samples[..., 0] - (self.scene_size[0])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 0] - (self.scene_size[1])).clamp(min=0, max=eps),) / eps
            v_weight = torch.min(torch.abs(samples[..., 1] - (self.scene_size[2])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 1] - (self.scene_size[3])).clamp(min=0, max=eps)) / eps
            w_weight = torch.min(torch.abs(samples[..., 2] - (self.scene_size[4])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 2] - (self.scene_size[5])).clamp(min=0, max=eps)) / eps
            weight = torch.stack([u_weight, v_weight, w_weight], dim=-1).detach()

            net_vel = weight * net_vel

        elif use_bdry_cond and self.cfg.src == 'vortex_collide':
            u_weight = torch.min(torch.abs(samples[..., 0] - (self.scene_size[0])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 0] - (self.scene_size[1])).clamp(min=0, max=eps),) / eps
            v_weight = torch.min(torch.abs(samples[..., 1] - (self.scene_size[2])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 1] - (self.scene_size[3])).clamp(min=0, max=eps)) / eps
            w_weight = torch.min(torch.abs(samples[..., 2] - (self.scene_size[4])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 2] - (self.scene_size[5])).clamp(min=0, max=eps)) / eps
            weight = torch.stack([u_weight, v_weight, w_weight], dim=-1).detach()

            net_vel = weight * net_vel

        elif use_bdry_cond and self.cfg.src == 'karman3d':
            inlet_mask = (samples[..., 2]>=self.scene_size[4]) & (samples[..., 2]<=self.scene_size[4]+eps)
            net_vel[..., 2][inlet_mask] = self.cfg.karman_vel

            dist = self.obs_sdf_func(samples)
            threshold = eps
            weight = torch.clamp(dist, 0, threshold) / threshold
            net_vel *= weight.unsqueeze(-1)

            # u_weight = torch.abs(samples[..., 0] - (self.scene_size[0])).clamp(min=0, max=eps) / eps
            u_weight = torch.min(torch.abs(samples[..., 0] - (self.scene_size[0])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 0] - (self.scene_size[1])).clamp(min=0, max=eps),) / eps
            v_weight = torch.min(torch.abs(samples[..., 1] - (self.scene_size[2])).clamp(min=0, max=eps),
                                torch.abs(samples[..., 1] - (self.scene_size[3])).clamp(min=0, max=eps)) / eps
            w_weight = torch.ones_like(samples[..., 2])
            weight = torch.stack([u_weight, v_weight, w_weight], dim=-1).detach()

            net_vel = weight * net_vel

        return net_vel

    def sample_in_training(self, resolution=None, apply_obstable=True):
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
        # if self.has_obstacle and apply_obstable:
        #     mask = (self.obs_sdf_func(samples) > -np.inf)
        #     samples = samples[mask]
        return samples

    def sample_velocity_field(self, resolution, to_numpy=True, with_boundary=True, return_samples=False, require_grad=False, flatten=True):
        grid_samples = sample_uniform_2D(resolution, with_boundary=with_boundary, size=self.scene_size, device=self.device)
        if flatten:
            grid_samples = grid_samples.reshape((-1, 3))
        if require_grad:
            grid_samples = grid_samples.requires_grad_(True)

        out = self.query_velocity(grid_samples, use_prev=True, use_bdry_cond=True)
        # out = self.velocity_field_prev(grid_samples)
        # if self.has_obstacle:
        #     mask = self.obs_sdf_func(grid_samples) <= 0
        #     out[mask] = 0

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


    ###################### Save Visualization TODO ################################################
    def draw_velocity(self, resolution, savepng, save_vdb = None):
        grid_values, grid_samples = self.sample_velocity_field(resolution, to_numpy=True, with_boundary=True, return_samples=True, flatten=False)
        print(grid_values.shape)

        if save_vdb is not None:
            # grid = vdb.FloatGrid()
            # grid.copyFromArray(grid_samples)
            vel = vdb.FloatGrid()
            vel.copyFromArray(np.linalg.norm(grid_values, axis=-1))
            vdb.write(save_vdb, grids=[vel])
            # np.savetxt(save_txt_v, grid_values)
            # np.savetxt(save_txt_s, grid_samples)
        grid_values = grid_values.reshape((-1, 3))
        grid_samples = grid_samples.reshape((-1, 3))
        draw_vector_field2D(grid_samples, grid_values, savepng, c=self.center, r=self.radius, figsize=self.fig_size)

    def draw_vorticity(self, resolution, save_txt_v = None, save_txt_s = None, vmin=-20, vmax=20):
        grid_values, grid_samples = self.sample_velocity_field(resolution, to_numpy=False, return_samples=True, require_grad=True)
        curl = curl2d(grid_values, grid_samples).squeeze(-1).detach().cpu().numpy()
        grid_samples = grid_samples.detach().cpu().numpy()

        if save_txt_v is not None:
            np.savetxt(save_txt_v, curl.reshape((curl.shape[0]*curl.shape[1], 1)))
            np.savetxt(save_txt_s, grid_samples.reshape((grid_samples.shape[0]*grid_samples.shape[1], grid_samples.shape[2])))

        # curl = (erf(curl) + 1) / 2 # map range to 0~1
        fig = draw_scalar_field2D(curl, vmin=vmin, vmax=vmax, figsize=self.fig_size, cmap='bwr', colorbar=False)
        # img = cm.bwr(curl)
        # img = Image.fromarray((img * 255).astype('uint8'))
        # if save_path is not None:
        #     img.save(save_path)
        return fig
        # fig = draw_scalar_field2D(curl, vmin=vmin, vmax=vmax, figsize=self.fig_size)
        # # x, y = grid_samples[..., 0].detach().cpu().numpy(), grid_samples[..., 1].detach().cpu().numpy()
        # # fig = draw_vorticity_field2D(curl, x, y)
        # return fig
    
    def draw_scalar(self, points, arr, savepng, save_txt_v = None, save_txt_s = None, vmin=None, vmax=None):
        if save_txt_v is not None:
            np.savetxt(save_txt_v, arr)
            np.savetxt(save_txt_s, points)
        draw_scalar_field2D(points, arr, savepng, figsize=self.fig_size)

    def compute_kinetic_energy(self, resolution):
        grid_values = self.sample_velocity_field(resolution, to_numpy=True, with_boundary=False)
        Ek = 0.5 * np.mean(grid_values ** 2)# + self.P
        return Ek


class NeuralFluidBase(NeuralFluidABC):
    def __init__(self, cfg):
        super(NeuralFluidBase, self).__init__(cfg)

    @NeuralFluidABC._training_loop
    def _add_source(self, source_func, is_init=True):
        """forward computation for add source"""
        # if (self.tb.train_iter == 0 or (self.tb.train_iter + 1) % self.cfg.vis_frequency == 0):
        #     self._vis_add_source_v(source_func, is_init)
        
        samples = self.sample_in_training(apply_obstable=False)

        out_rand = self.query_velocity(samples, use_bdry_cond=True)
        if is_init:
            target_rand_val = source_func(samples)
        else:
            raise NotImplemented

        loss_random = F.mse_loss(out_rand, target_rand_val)
        loss_dict = {'main': loss_random}
        # if self.cfg.grad_sup:
        #     target_rand_grad = jacobian(target_rand_val, samples)[0][..., [0, 1], [0, 1]]
        #     out_grad = jacobian(out_rand, samples)[0][..., [0, 1], [0, 1]]
        #     grad_loss = F.mse_loss(out_grad, target_rand_grad)
        #     loss_dict.update({"grad_mse": grad_loss})

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
    
    def smoothstep_circular_obs(self, samples, vel, eps=1e-1):
        # d = self.obs_sdf_func(samples)
        # mask = (d < eps) & (d > 0)
        # inner_mask = (d < 0)
        
        # l = torch.ones_like(d, device=self.device)
        # # l[mask] = self._smoothstep_linear(d[mask], 0.0, eps)
        # l[inner_mask] = 0.0
        # vel[..., 0] *= l
        # vel[..., 1] *= l

        dist = self.obs_sdf_func(samples)
        threshold = eps
        weight = torch.clamp(dist, 0, threshold) / threshold
        vel *= weight.unsqueeze(-1)

        return vel
    
    # def smoothstep_circular_obs(self, samples, vel, eps=1e-1):
    #     d = self.obs_sdf_func(samples)
    #     n_ortho = torch.zeros_like(vel)
    #     n_ortho[..., 0] = -(self.center[0][1]-samples[..., 1])/(d+self.radius[0])
    #     n_ortho[..., 1] = (self.center[0][0]-samples[..., 0])/(d+self.radius[0])

    #     mask = (d <= eps)
    #     inner_mask = (d < 0)
        
    #     l = self._smoothstep_poly(d[mask], 0.0, eps)
    #     vel_dot_n = torch.sum(vel[mask].clone() * n_ortho[mask], dim=-1)
    #     vel[..., 0][mask] = n_ortho[..., 0][mask] * vel_dot_n * l
    #     vel[..., 1][mask] = n_ortho[..., 1][mask] * vel_dot_n * l
    #     vel[inner_mask] = vel[inner_mask] * 0.0

    #     return vel

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