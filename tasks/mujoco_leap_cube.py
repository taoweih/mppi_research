import mujoco
import mujoco.rollout
import mujoco.viewer

import numpy as np
import torch
import warnings

from pathlib import Path
import sys
sys.path.append("..")
import custom_mppi
import base_mppi
import time

from judo.utils.math_utils import quat_diff_so3

# from judo mpc cylinder push task
ROOT = Path(__file__).resolve().parent
model_xml_path = str(ROOT/"models/judo_mpc_models/xml/leap_cube.xml")

NUM_THREAD = 16
MODEL = mujoco.MjModel.from_xml_path(model_xml_path)

model = MODEL
data = mujoco.MjData(model)
model.opt.timestep = 0.01

data_list = [mujoco.MjData(model) for _ in range(NUM_THREAD)]

TIMESTEPS = 100  # T
N_SAMPLES = 200  # K
ACTION_LOW = -10.0
ACTION_HIGH = 10.0

nx = model.nq + model.nv # +1 since time is considered a state my mujoco


d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if d == torch.device("cpu"):
    d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if d == torch.device("cpu"):
    warnings.warn("No GPU device detected, using cpu instead", UserWarning)
dtype = torch.float32

noise_sigma = 0.1*torch.eye(model.nu, device=d, dtype=dtype)
lambda_ = 1.

def dynamics(state, perturbed_action):
    #load state into a MjData instance of the model
    control = perturbed_action.unsqueeze(1).cpu().numpy()
    state, _ = mujoco.rollout.rollout(model=model, data=data_list, initial_state = state.cpu().numpy(), nstep = 1, control=control, persistent_pool=True)
    return torch.tensor(state).squeeze(1) # remove the nsteps dimension since it's only simulated for 1 time step

goal_pos = np.array([0.0,0.03,0.1])
goal_quat = np.array([1.0,0.0,0.0,0.0]) #default goal_quat
w_pos = 100.0
w_rot = 0.1

def running_cost(state,action):
    cost = 0

    qo_pos_traj = state[..., :3]
    qo_quat_traj = state[..., 3:7]
    qo_pos_diff = qo_pos_traj - goal_pos
    qo_quat_diff = quat_diff_so3(qo_quat_traj, goal_quat)

    pos_cost = w_pos * 0.5 * np.square(qo_pos_diff).sum(-1).mean(-1)
    rot_cost = w_rot * 0.5 * np.square(qo_quat_diff).sum(-1).mean(-1)
    cost += pos_cost + rot_cost

    return torch.tensor(cost, dtype=dtype, device=d)

def terminal_cost(state):
    cost = 0
    return cost

mppi = custom_mppi.CUSTOM_MPPI(dynamics, running_cost, nx, noise_sigma, use_mujoco_physics=False, terminal_cost = terminal_cost, num_samples=N_SAMPLES, time_steps=TIMESTEPS, steps_per_stage=20,
                         lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
                         u_max=torch.tensor(ACTION_HIGH, device=d), device=d)
# initilization
qpos_home = np.array(
    [
        0.0, 0.03, 0.1, 1.0, 0.0, 0.0, 0.0,  # cube
        0.5, -0.75, 0.75, 0.25,  # index
        0.5, 0.0, 0.75, 0.25,  # middle
        0.5, 0.75, 0.75, 0.25,  # ring
        0.65, 0.9, 0.75, 0.6,  # thumb
    ]
) 
reset_command = np.array(
            [
                0.5, -0.75, 0.75, 0.25,  # index
                0.5, 0.0, 0.75, 0.25,  # middle
                0.5, 0.75, 0.75, 0.25,  # ring
                0.65, 0.9, 0.75, 0.6,  # thumb
            ]
        ) 

data.qpos[:] = qpos_home
data.qvel[:] = 0.0
data.ctrl[:] = reset_command
uvw = np.random.rand(3)
goal_quat = np.array(
            [
                np.sqrt(1 - uvw[0]) * np.sin(2 * np.pi * uvw[1]),
                np.sqrt(1 - uvw[0]) * np.cos(2 * np.pi * uvw[1]),
                np.sqrt(uvw[0]) * np.sin(2 * np.pi * uvw[2]),
                np.sqrt(uvw[0]) * np.cos(2 * np.pi * uvw[2]),
            ]
        )
data.mocap_quat[0] = goal_quat

mujoco.mj_forward(model,data)


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # # vectorize robot states
        state = np.concatenate([[data.time],data.qpos, data.qvel])
        state = torch.tensor(state, dtype = dtype, device = d)

        # now = time.time()
        action, _, _ = mppi.command(state)
        # print(f"time: {time.time() - now}")
 
        data.ctrl[:] = action.cpu().numpy()
        mujoco.mj_step(model,data)
        viewer.sync()


