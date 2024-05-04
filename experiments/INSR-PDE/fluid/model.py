import os
import numpy as np
import torch
import torch.nn.functional as F
from base import BaseModel, sample_random, sample_uniform, sample_boundary2D_separate
from base import gradient, divergence, laplace, jacobian
from .examples import get_examples
from .visualize import draw_vector_field2D, draw_scalar_field2D, draw_curl, draw_magnitude, save_numpy_img, save_figure


class Fluid2DModel(BaseModel):
    """inviscid Navier-Stokes equation. 2D fluid. [-1, 1]^2."""
    def __init__(self, cfg):
        super().__init__(cfg)

        self.velocity_field = self._create_network(2, 2)
        self.velocity_field_prev = self._create_network(2, 2)
        self.pressure_field = self._create_network(2, 1)
        self._set_require_grads(self.velocity_field_prev, False)

    @property
    def _trainable_networks(self):
        return {'velocity': self.velocity_field, 'pressure': self.pressure_field}
    
    def _sample_in_training(self):
        return sample_random(self.sample_resolution ** 2, 2, device=self.device).requires_grad_(True)

    def sample_field(self, resolution, return_samples=False):
        """sample current field with uniform grid points"""
        grid_samples = sample_uniform(resolution, 2, device=self.device, flatten=False).requires_grad_(True)
        out = self.velocity_field(grid_samples)
        if return_samples:
            return out, grid_samples
        return out

    @BaseModel._timestepping
    def initialize(self):
        if not hasattr(self, "init_cond_func"):
            self.init_cond_func = get_examples(self.cfg.init_cond)
        self._initialize()

    @BaseModel._training_loop
    def _initialize(self):
        """forward computation for initialization"""
        samples = self._sample_in_training()
        ref = self.init_cond_func(samples)
        out = self.velocity_field(samples)
        loss_random = F.mse_loss(out, ref)

        loss_dict = {'main': loss_random}
        return loss_dict
    
    def _vis_initialize(self):
        """visualization on tb during training"""
        velos, samples = self.sample_field(self.vis_resolution, return_samples=True)
        velos = velos.detach().cpu().numpy()
        samples = samples.detach().cpu().numpy()
        fig = draw_vector_field2D(velos, samples)
        self.tb.add_figure("velocity", fig, global_step=self.train_step)

    @BaseModel._timestepping
    def step(self):
        """operater splitting scheme"""
        self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())
        self._advect_velocity()

        self._solve_pressure()

        self.velocity_field_prev.load_state_dict(self.velocity_field.state_dict())
        self._projection()
    
    @BaseModel._training_loop
    def _advect_velocity(self):
        """velocity advection: dudt = -(u\cdot grad)u"""
        samples = self._sample_in_training()

        # dudt
        with torch.no_grad():
            prev_u = self.velocity_field_prev(samples).detach()
        curr_u = self.velocity_field(samples)

        # backtracking
        backtracked_position = samples - prev_u * self.cfg.dt
        backtracked_position = torch.clamp(backtracked_position, min=-1.0, max=1.0)
        
        with torch.no_grad():
            advected_u = self.velocity_field_prev(backtracked_position).detach()

        loss = torch.mean((curr_u - advected_u) ** 2)
        loss_dict = {'main': loss}

        # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
        #        and fixed factor 1.0 for boundary loss
        bc_sample_x = sample_boundary2D_separate(samples.shape[0] // 100, side='horizontal', device=self.device).requires_grad_(True)
        bc_sample_y = sample_boundary2D_separate(samples.shape[0] // 100, side='vertical', device=self.device).requires_grad_(True)
        vel_x = self.velocity_field(bc_sample_x)[..., 0]
        vel_y = self.velocity_field(bc_sample_y)[..., 1]
        bc_loss = (torch.mean(vel_x ** 2) + torch.mean(vel_y ** 2)) * 1.0
        loss_dict.update({"bc": bc_loss})

        return loss_dict

    @BaseModel._training_loop
    def _solve_pressure(self):
        """solve pressure: div u = lap P."""
        samples = self._sample_in_training()

        out_u = self.velocity_field(samples)
        div_u = divergence(out_u, samples).detach()
        out_p = self.pressure_field(samples)
        lap_p = laplace(out_p, samples)

        loss = torch.mean((div_u - lap_p) ** 2) # FIXME: assume rho=1 here
        loss_dict = {'main': loss}

        # NOTE: neumann boundary condition, grad(p)\cdot norm(p) = 0
        bc_sample_x = sample_boundary2D_separate(self.sample_resolution ** 2 // 100, side='horizontal', device=self.device).requires_grad_(True)
        bc_sample_y = sample_boundary2D_separate(self.sample_resolution ** 2 // 100, side='vertical', device=self.device).requires_grad_(True)
        grad_px = gradient(self.pressure_field(bc_sample_x), bc_sample_x)[..., 0]
        grad_py = gradient(self.pressure_field(bc_sample_y), bc_sample_y)[..., 1]

        bc_loss = torch.mean(grad_px ** 2) + torch.mean(grad_py ** 2)
        loss_dict.update({'bc': bc_loss})

        return loss_dict

    @BaseModel._training_loop
    def _projection(self):
        """velocity projection: u <- u - grad(p)"""
        samples = self._sample_in_training()

        with torch.no_grad():
            prev_u = self.velocity_field_prev(samples).detach()
        
        p = self.pressure_field(samples)
        grad_p = gradient(p, samples).detach()

        target_u = prev_u - grad_p
        curr_u = self.velocity_field(samples)
        loss = torch.mean((curr_u - target_u) ** 2)
        loss_dict = {'main': loss}

        # FIXME: hard-coded zero boundary condition to sample 1% points near boundary
        #        and fixed factor 1.0 for boundary loss
        bc_sample_x = sample_boundary2D_separate(samples.shape[0] // 100, side='horizontal', device=self.device).requires_grad_(True)
        bc_sample_y = sample_boundary2D_separate(samples.shape[0] // 100, side='vertical', device=self.device).requires_grad_(True)
        vel_x = self.velocity_field(bc_sample_x)[..., 0]
        vel_y = self.velocity_field(bc_sample_y)[..., 1]
        bc_loss = (torch.mean(vel_x ** 2) + torch.mean(vel_y ** 2)) * 1.0
        loss_dict.update({"bc": bc_loss})
        return loss_dict

    def _vis_advect_velocity(self):
        """visualization on tb during training"""
        curr_u_grid, grid_samples = self.sample_field(self.vis_resolution, return_samples=True)
        with torch.no_grad():
            prev_u_grid = self.velocity_field_prev(grid_samples).detach()

        backtracked_position = grid_samples - prev_u_grid * self.cfg.dt
        backtracked_position = torch.clamp(backtracked_position, min=-1.0, max=1.0)
        
        with torch.no_grad():
            advected_u = self.velocity_field_prev(backtracked_position).detach()

        mse = torch.mean((curr_u_grid - advected_u) ** 2, dim=-1).detach().cpu().numpy()
        curr_u_grid = curr_u_grid.detach().cpu().numpy()
        grid_samples = grid_samples.detach().cpu().numpy()
        self.tb.add_figure('adv_mse', draw_scalar_field2D(mse), global_step=self.train_step)
        self.tb.add_figure('adv_u', draw_vector_field2D(curr_u_grid, grid_samples), global_step=self.train_step)

    def _vis_solve_pressure(self):
        """visualization on tb during training"""
        out_u, grid_samples = self.sample_field(self.vis_resolution, return_samples=True)
        div_u = divergence(out_u, grid_samples).detach()
        out_p = self.pressure_field(grid_samples)
        lap_p = laplace(out_p, grid_samples)
        grad_p = gradient(out_p, grid_samples)
        mse = (div_u - lap_p) ** 2

        self.tb.add_figure('pre_div', draw_scalar_field2D(div_u[..., 0].detach().cpu().numpy()), global_step=self.train_step)
        self.tb.add_figure('pre_p_lap', draw_scalar_field2D(lap_p[..., 0].detach().cpu().numpy()), global_step=self.train_step)
        self.tb.add_figure('pre_p', draw_scalar_field2D(out_p[..., 0].detach().cpu().numpy()), global_step=self.train_step)
        self.tb.add_figure('pre_p_gradx', draw_scalar_field2D(grad_p[..., 0].detach().cpu().numpy()), global_step=self.train_step)
        self.tb.add_figure('pre_p_grady', draw_scalar_field2D(grad_p[..., 1].detach().cpu().numpy()), global_step=self.train_step)
        self.tb.add_figure('pre_mse', draw_scalar_field2D(mse[..., 0].detach().cpu().numpy()), global_step=self.train_step)

    def _vis_projection(self):
        """visualization on tb during training"""
        curr_u, grid_samples = self.sample_field(self.vis_resolution, return_samples=True)

        with torch.no_grad():
            prev_u = self.velocity_field_prev(grid_samples).detach()
        p = self.pressure_field(grid_samples)
        grad_p = gradient(p, grid_samples).detach()
        target_u = prev_u - grad_p
        mse = torch.sum((curr_u - target_u) ** 2, dim=-1).detach().cpu().numpy()

        grad_p = grad_p.detach().cpu().numpy()
        target_u = target_u.detach().cpu().numpy()
        curr_u = curr_u.detach().cpu().numpy()
        grid_samples = grid_samples.detach().cpu().numpy()
        self.tb.add_figure('proj_grad_p', draw_vector_field2D(grad_p, grid_samples), global_step=self.train_step)
        self.tb.add_figure('proj_target_u', draw_vector_field2D(target_u, grid_samples), global_step=self.train_step)
        self.tb.add_figure('proj_out_u', draw_vector_field2D(curr_u, grid_samples), global_step=self.train_step)
        self.tb.add_figure('proj_mse', draw_scalar_field2D(mse), global_step=self.train_step)

    def write_output(self, output_folder):
        grid_u, grid_samples = self.sample_field(100, return_samples=True)

        # u_mag = torch.sqrt(torch.sum(grid_u ** 2, dim=-1))
        # jaco, _ = jacobian(grid_u, grid_samples)
        # u_curl = jaco[..., 1, 0] - jaco[..., 0, 1]
        
        grid_samples = grid_samples.detach().cpu().numpy()
        grid_u = grid_u.detach().cpu().numpy()
        # u_mag = u_mag.detach().cpu().numpy()
        # u_curl = u_curl.detach().cpu().numpy()

        fig = draw_vector_field2D(grid_u, grid_samples)
        save_path = os.path.join(output_folder, f"t{self.timestep:03d}_vel.png")
        save_figure(fig, save_path)

        # mag_img = draw_magnitude(u_mag)
        # save_path = os.path.join(output_folder, f"t{self.timestep:03d}_mag.png")
        # save_numpy_img(mag_img, save_path)

        # curl_img = draw_curl(u_curl)
        # save_path = os.path.join(output_folder, f"t{self.timestep:03d}_curl.png")
        # save_numpy_img(curl_img, save_path)

        save_path = os.path.join(output_folder, f"t{self.timestep:03d}.npy")
        np.save(save_path, grid_u)
