import os

import torch
import torch.nn.functional as F

def positional_constraint_loss(q_fixed, q_fixed_target, ratio_constraint):
	E_positional = ratio_constraint * torch.sum((q_fixed - q_fixed_target) ** 2) 
	return E_positional

def collision_plane_loss(q, qdot, dt, ratio_collide, plane_height):
	'''loss when the shape is collide with a plane'''
	E_collide = 0
	collide_indices = (q[:, -1] < plane_height)
	q_collide = q[collide_indices, :]
	if q_collide.shape[0] > 0:
		qdot_collide = qdot[collide_indices, :]
		collide_dist = plane_height - q_collide[:, -1]
		collide_force = ratio_collide * torch.column_stack((torch.zeros(collide_dist.shape[0], q.shape[1]-1).cuda(), collide_dist))
		E_collide = -dt * torch.sum(torch.mul(qdot_collide, collide_force))
	return E_collide

def collision_sphere_loss(q, qdot, dt, ratio_collide, circle_center, circle_radius):
	'''loss when the shape is collide with a sphere'''
	E_collide = 0
	collide_vec = q - circle_center
	collide_dist = torch.sqrt(torch.sum(collide_vec ** 2, dim = 1))
	collide_dir = collide_vec / collide_dist[:, None]
	collide_indices = (collide_dist < circle_radius)
	q_collide = q[collide_indices, :]
	if q_collide.shape[0] > 0:
		qdot_collide = qdot[collide_indices, :]
		collide_dist = collide_dist[collide_indices]
		collide_dir = collide_dir[collide_indices, :]
		if q.shape[1] == 2:
			collide_force = ratio_collide * collide_dist[:, None] * collide_dir
		else:
			collide_force = ratio_collide * collide_dist[:, None, None] * collide_dir
		E_collide = -dt * torch.sum(torch.mul(qdot_collide, collide_force))
	return E_collide

def collision_bowl_loss(q, qdot, dt, ratio_collide, circle_center, circle_radius):
	'''loss when the shape is collide with a bowl (i.e., bottom half of a circle)'''
	E_collide = 0
	collide_vec = circle_center - q
	collide_dist = torch.sqrt(torch.sum(collide_vec ** 2, dim = 1))
	collide_dir = collide_vec / collide_dist[:, None]
	collide_indices = ((collide_dist > circle_radius) & (q[:, 2] < circle_center[2]))
	q_collide = q[collide_indices, :]
	if q_collide.shape[0] > 0:
		qdot_collide = qdot[collide_indices, :]
		collide_dist = collide_dist[collide_indices]
		collide_dir = collide_dir[collide_indices, :]
		if q.shape[1] == 2:
			collide_force = ratio_collide * collide_dist[:, None] * collide_dir
		else:
			collide_force = ratio_collide * collide_dist[:, None, None] * collide_dir
		E_collide = -dt * torch.sum(torch.mul(qdot_collide, collide_force))
	return E_collide

def collision_bar_loss(q, qdot, dt, ratio_collide, bar_height, bar_width, bar_num):
	'''loss when the shape is collide with a sphere'''
	E_collide = 0
	return E_collide
