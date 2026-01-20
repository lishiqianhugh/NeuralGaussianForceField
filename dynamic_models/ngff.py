import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from dynamic_models import *
from tools.monitor import timeit, memoryit

class NGFFDynamics(nn.Module):
    def __init__(self, predictor, boundary, mass, gravity, bs, num_objs, N, feature_dim, ori_pos, external_forces):
        super(NGFFDynamics, self).__init__()
        # Register predictor so its parameters are tracked for adjoint method
        self.predictor = predictor
        self.boundary = boundary
        self.mass = mass
        self.gravity = gravity
        self.bs = bs
        self.num_objs = num_objs
        self.N = N
        self.FEATURE_DIM = feature_dim
        self.ori_pos = ori_pos
        self.external_forces = external_forces

    def forward(self, t, y):
        bs = self.bs
        num_objs = self.num_objs
        N = self.N
        FEATURE_DIM = self.FEATURE_DIM

        points = y[:, :, :N*FEATURE_DIM]
        points = points.reshape(bs, num_objs, N, FEATURE_DIM)
        points_vel = y[:, :, N*FEATURE_DIM:2*N*FEATURE_DIM]
        points_vel = points_vel.reshape(bs, num_objs, N, FEATURE_DIM)
        com = y[:, :, 2*N*FEATURE_DIM: 2*N*FEATURE_DIM + 3]
        com_vel = y[:, :, 2*N*FEATURE_DIM + 3: 2*N*FEATURE_DIM + 6]
        angle = y[:, :, 2*N*FEATURE_DIM + 6: 2*N*FEATURE_DIM + 9]
        angle_vel = y[:, :, 2*N*FEATURE_DIM + 9: 2*N*FEATURE_DIM + 12]
        padding = y[:, :, 2*N*FEATURE_DIM + 12:2*N*FEATURE_DIM + 12 + N]
        knn = y[:, :, 2*N*FEATURE_DIM + 12 + N:]
        knn = knn.reshape(bs, num_objs, N, -1)

        forces, torques, stresses = self.predictor(points, self.ori_pos, points_vel, com, angle, com_vel, angle_vel, padding, knn, self.boundary)
        com_acc = forces / self.mass
        if self.external_forces:
            for i in range(len(self.external_forces)):
                if t > self.external_forces[i]['start'] and t < self.external_forces[i]['end']:
                    if 'force' in self.external_forces[i]:
                        com_acc[:,self.external_forces[i]['obj_id']] += self.external_forces[i]['force'] / self.mass
                    if 'torque' in self.external_forces[i]:
                        torques[:,self.external_forces[i]['obj_id']] += self.external_forces[i]['torque'] / self.mass
        com_acc[..., 2] += self.gravity
        angle_acc = torques / self.mass
        points_derivative = points_vel
        points_derivative = points_derivative * padding.unsqueeze(-1)
        points_derivative = points_derivative.reshape(bs, num_objs, N*FEATURE_DIM)
        stresses = stresses * padding.unsqueeze(-1)
        stresses = stresses.reshape(bs, num_objs, N*FEATURE_DIM)
        knn = knn.reshape(bs, num_objs, -1)
        return torch.cat((points_derivative, 
                          stresses,
                          com_vel, 
                          com_acc,
                          angle_vel,
                          angle_acc,
                          torch.zeros_like(padding),
                          torch.zeros_like(knn, dtype=torch.long)), dim=-1)

class NGFFobj(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=3, num_layers=6, num_keypoints=1024, k=8, mass=1.0, dt=0.01, ode_method='euler', r=8, step_size=1e-3, threshold=1e-2, rtol=1e-3, atol=1e-4):
        super(NGFFobj, self).__init__()
        self.predictor = IN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, threshold=threshold, num_keypoints=num_keypoints, k=k)
        self.mass = mass
        self.dt = dt
        self.ode_method = ode_method  # 'adaptive' or 'euler'
        self.step_size = step_size
        self.rtol = rtol
        self.atol = atol
        self.gravity = nn.Parameter(torch.tensor(-5.0), requires_grad=False)
        # self.boundary = torch.tensor([0.94, -0.94, 0.94, -0.94, 0.94, -0.94])
        self.boundary = torch.tensor([1.94, -1.94, 1.94, -1.94, 2.94, -0.94])

    def forward(self, points, com, com_vel, angle, angle_vel, padding, knn, pred_len=10, external_forces=None):
        bs, num_objs, N, FEATURE_DIM = points.shape
        points = points - com.unsqueeze(2)  # Center points around the center of mass
        ori_pos = points.clone()  # (B, num_objs, N, 3)
        points = points.reshape(bs, num_objs, -1) # (B, num_objs, N*3)
        knn = knn.reshape(bs, num_objs, -1)  # (B, num_objs, N*K)

        # Define ODE dynamics nn.Module
        func = NGFFDynamics(
            predictor=self.predictor,
            boundary=self.boundary,
            mass=self.mass,
            gravity=self.gravity,
            bs=bs,
            num_objs=num_objs,
            N=N,
            feature_dim=FEATURE_DIM,
            ori_pos=ori_pos,
            external_forces=external_forces,
        )

        # Initial state
        points_vel = torch.zeros_like(points)  # Initialize points velocity to zero
        y0 = torch.cat([points, points_vel, com, com_vel, angle, angle_vel, padding, knn], dim=-1)  # Shape: (batch_size, N, 6)
        t = torch.linspace(0, pred_len * self.dt, pred_len + 1).to(points.device)  # Time grid
        if self.ode_method == 'euler':
            solution = odeint(func, y0, t, method='euler', options={'step_size':self.step_size})  # Solution will be (pred_len+1, batch_size, N, 6)
        elif self.ode_method == 'adaptive':
            solution = odeint(func, y0, t, rtol=self.rtol, atol=self.atol)
        solution = solution.transpose(0, 1)  # Swap the first two dimensions (batch_size, pred_len+1, N, 6)
        
        points_trajectory = solution[:, :, :, :N*FEATURE_DIM] # (B, pred_len+1, num_objs, N*3)
        points_trajectory = points_trajectory.reshape(bs, pred_len+1, num_objs, N, FEATURE_DIM)
        com_trajectory = solution[:, :, :, 2*N*FEATURE_DIM: 2*N*FEATURE_DIM + 3]
        angle_trajectory = solution[:, :, :, 2*N*FEATURE_DIM + 6: 2*N*FEATURE_DIM + 9]

        return points_trajectory, com_trajectory, angle_trajectory
    