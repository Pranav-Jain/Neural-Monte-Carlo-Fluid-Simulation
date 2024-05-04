import os
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import shutil
from tensorboardX import SummaryWriter
from networks import PIDeepONet
from utils.diff_ops import jacobian, divergence, curl2d_fdiff, gradient
from utils.model_utils import sample_uniform_2D, sample_random_2D, sample_boundary_separate
from utils.vis_utils import draw_scalar_field2D, draw_vector_field2D


class NeuralFluid(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg.dt
        self.t_range = cfg.t_range
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
        self.device = torch.device("cuda:0")

        self.loss_record = [10000, 0] # for early stopping condition

        # neural implicit network for density, velocity and pressure field
        # n_field_dim = 1
        self.n_branch_in = 100
        n_trunk_in = 3
        n_out = 60
        num_hidden_layers = cfg.num_hidden_layers
        hidden_features = cfg.hidden_features
        nonlinearity = "sine"
        
        self.field = PIDeepONet(3, self.n_branch_in * 2, n_trunk_in, n_out, 
            num_hidden_layers, hidden_features, nonlinearity).to(self.device)
        
        self.x0 = sample_uniform_2D(int(np.sqrt(self.n_branch_in)), 2, device=self.device)
        self.x0 = self.x0.view(-1, 2)
    
    @property
    def _trainable_networks(self):
        return {'field': self.field}

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def create_optimizer(self, use_scheduler=True, gamma=0.1, patience=5000, min_lr=1e-8):
        self.loss_record = [10000, 0]
        # optimizer: use only one optimizer?
        param_list = []
        for net in self._trainable_networks.values():
            param_list.append({"params": net.parameters(), "lr": self.cfg.lr})
        self.optimizer = torch.optim.Adam(param_list)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=gamma, 
        #     min_lr=min_lr, patience=patience, verbose=True) if use_scheduler else None
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95 ** 0.0001)
        # self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=1e-7, max_lr=self.cfg.lr, 
        #     step_size_up=1000, step_size_down=1000, mode='triangular2', cycle_momentum=False) if use_scheduler else None

    def create_tb(self, name, overwrite=True):
        """create tensorboard log"""
        self.log_path = os.path.join(self.cfg.log_dir, name)
        if os.path.exists(self.log_path) and overwrite:
            shutil.rmtree(self.log_path, ignore_errors=True)
        return SummaryWriter(self.log_path)

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip > 0:
            param_list = []
            for net in self._trainable_networks.values():
                param_list = param_list + list(net.parameters())
            torch.nn.utils.clip_grad_norm_(param_list, 0.1)
        self.optimizer.step()
        if self.scheduler is not None:
            # self.scheduler.step(loss_dict['main'])
            self.scheduler.step()
        self.update_loss_record(loss_dict['main'].item())
    
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
    
    def load_ckpt(self, name=None, path=None):
        """load saved checkpoint"""
        if path is not None:
            load_path = path
        else:
            if type(name) is int:
                load_path = os.path.join(self.cfg.model_dir, f"ckpt_step_t{name:03d}.pth")
            else:
                load_path = os.path.join(self.cfg.model_dir, f"ckpt_{name}.pth")
        checkpoint = torch.load(load_path)

        for name, net in self._trainable_networks.items():
            net.load_state_dict(checkpoint[f'net_{name}'])
        self.timestep = checkpoint['timestep']

    def set_source(self, source_func):
        self.v0 = source_func(self.x0).squeeze(-1).reshape(-1) # (N, 2)
        print("v0", self.v0.shape)

    def train(self, source_func):
        if not hasattr(self, 'v0'):
            self.set_source(source_func)

        pbar = tqdm(range(self.max_n_iters))
        self.create_optimizer()
        self.tb = self.create_tb("train")
        for i in pbar:
            loss_dict = self._train_step(source_func)
            self.update_network(loss_dict)

            loss_value = {k: v.item() for k, v in loss_dict.items()}
            self.tb.add_scalars("loss", loss_value, global_step=i)
            pbar.set_postfix(loss_value)

            if i == 0 or (i + 1) % self.cfg.vis_frequency == 0:
                vis_res = 32
                fig = self.draw_velocity(vis_res, 0)
                self.tb.add_figure("t=0", fig, global_step=i)
                fig = self.draw_velocity(vis_res, self.t_range // 2)
                self.tb.add_figure(f"t={self.t_range // 2}", fig, global_step=i)
                fig = self.draw_velocity(vis_res, self.t_range)
                self.tb.add_figure(f"t={self.t_range}", fig, global_step=i)
                lr = self.get_lr()
                self.tb.add_scalar("lr", lr, global_step=i)
            
            if i % (self.max_n_iters // 10) == 0:
                self.save_ckpt(str(i))
            
            if self.cfg.early_stop and self.optimizer.param_groups[0]['lr'] <= 1.1e-8 and self.loss_record[1] >= 500:
                pbar.write(f"early stopping at iteration {i}")
                break
        self.save_ckpt("final")

    def _train_step(self, source_func):
        # initial condition
        x_init, _ = self.sample_in_training(is_init=True)
        t_init = torch.zeros(x_init.shape[:-1], device=self.device).unsqueeze(-1)

        u_init = self.field(self.v0, x_init, t_init)[..., :2]
        u_init_gt = source_func(x_init)

        loss_init = F.mse_loss(u_init, u_init_gt)

        # boundary condition
        n_bc_samples = x_init.shape[0] // 100
        bc_sample_x = sample_boundary_separate(n_bc_samples, side='horizontal', device=self.device).requires_grad_(True)
        bc_sample_y = sample_boundary_separate(n_bc_samples, side='vertical', device=self.device).requires_grad_(True)
        t_bound = torch.rand(bc_sample_x.shape[0], device=self.device).unsqueeze(-1) * self.t_range # (0, t_range)
        vel_x = self.field(self.v0, bc_sample_x, t_bound)[..., 0]
        vel_y = self.field(self.v0, bc_sample_y, t_bound)[..., 1]
        loss_bound = (torch.mean(vel_x ** 2) + torch.mean(vel_y ** 2)) * 1.0

        # pde residual
        x_main, t_main = self.sample_in_training(is_init=False)
        # t_main = torch.rand(self.sample_resolution ** 2, device=self.device).unsqueeze(-1) * self.t_range # (0, t_range)
        x_main.requires_grad_(True)
        t_main.requires_grad_(True)
        out = self.field(self.v0, x_main, t_main)
        u_main = out[..., :2] # (N, 2)
        p_main = out[..., -1] # (N, )

        # div(u) = 0
        div_u = divergence(u_main, x_main)
        loss_divU = torch.mean(div_u ** 2)

        # du/dt + u \dot grad(u) + grap(p) = 0
        dudt = jacobian(u_main, t_main)[0].squeeze(-1)
        jac_u, status = jacobian(u_main, x_main)
        if status == -1:
            raise RuntimeError("jacobian has NaN")
        dudx = jac_u[..., 0] # gradient(u_main, x_main[..., 0])
        dudy = jac_u[..., 1] # gradient(u_main, x_main[..., 1])
        grad_p = gradient(p_main, x_main)
        pde = dudt + u_main[..., :1] * dudx + u_main[..., 1:] * dudy + grad_p
        loss_main = torch.mean(pde ** 2)

        loss_dict = {"init": loss_init, "bound": loss_bound, "main": loss_main, "div": loss_divU}
        return loss_dict

    def sample_in_training(self, is_init=False):
        if self.sample_pattern == 'random':
            samples = sample_random_2D(self.sample_resolution ** 2, device=self.device).requires_grad_(True)
            time = torch.rand(self.sample_resolution ** 2, device=self.device).unsqueeze(-1) * self.t_range # (0, t_range)
            return samples, time
        elif self.sample_pattern == 'uniform':
            samples = sample_uniform_2D(self.sample_resolution, device=self.device).requires_grad_(True)
        elif self.sample_pattern == 'random+uniform':
            samples = torch.cat([sample_random_2D(self.sample_resolution ** 2, device=self.device),
                        sample_uniform_2D(self.sample_resolution, device=self.device).view(-1, 2)], dim=0).requires_grad_(True)
        elif self.sample_pattern == 'fixed':
            n = 40960000
            if not hasattr(self, "pre_samples"):
                self.pre_samples = sample_random_2D(n, device=self.device)
                self.pre_samples_t = torch.rand(n, device=self.device).unsqueeze(-1) * self.t_range
                self.pre_samples_init = sample_random_2D(n, device=self.device)
            indices = torch.randint(0, n - 1, size=(self.sample_resolution ** 2, ), device=self.device)
            if is_init:
                return self.pre_samples_init[indices], None
            else:
                return self.pre_samples[indices], self.pre_samples_t[indices]
        else:
            raise NotImplementedError
        return samples, None

    def sample_velocity_field(self, resolution, t, to_numpy=True, with_boundary=False, return_samples=False, require_grad=False):
        grid_samples = sample_uniform_2D(resolution, with_boundary=with_boundary, device=self.device).view(-1, 2)
        time = torch.ones(grid_samples.shape[:-1], device=self.device).unsqueeze(-1) * t
        if require_grad:
            grid_samples = grid_samples.requires_grad_(True)

        out = self.field(self.v0, grid_samples, time)[..., :2]
        out = out.view(resolution, resolution, 2)
        grid_samples = grid_samples.view(resolution, resolution, 2)
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
            print(f"no method named '{func_str}'.")
            pass

    def draw_velocity(self, resolution, t):
        grid_values, grid_samples = self.sample_velocity_field(resolution, t, to_numpy=True, return_samples=True)
        x, y = grid_samples[..., 0], grid_samples[..., 1]
        fig = draw_vector_field2D(grid_values[..., 0], grid_values[..., 1], x, y)
        return fig
