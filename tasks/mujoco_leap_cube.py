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

import imageio
from tqdm import tqdm

from judo.utils.math_utils import quat_diff_so3

# from judo mpc cylinder push task
ROOT = Path(__file__).resolve().parent
model_xml_path = str(ROOT/"models/judo_mpc_models/xml/leap_cube.xml")

NUM_THREAD = 30
MODEL = mujoco.MjModel.from_xml_path(model_xml_path)

model = MODEL
data = mujoco.MjData(model)
model.opt.timestep = 0.001
sim_model = mujoco.MjModel.from_xml_path(model_xml_path)
sim_model.opt.timestep = 0.02


renderer = mujoco.Renderer(model,height=480, width=640)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.azimuth = 225
cam.elevation = -45
cam.distance = 1
cam.lookat[:] = [0,0,0]

data_list = [mujoco.MjData(model) for _ in range(NUM_THREAD)]

TIMESTEPS = 5  # T
N_SAMPLES = 200  # K
ACTION_LOW = torch.tensor(model.actuator_ctrlrange[:,0])
ACTION_HIGH = torch.tensor(model.actuator_ctrlrange[:,1])

nx = model.nq + model.nv # +1 since time is considered a state my mujoco


d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if d == torch.device("cpu"):
    d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if d == torch.device("cpu"):
    warnings.warn("No GPU device detected, using cpu instead", UserWarning)
dtype = torch.float32

noise_sigma = (torch.max(ACTION_HIGH.abs(),ACTION_LOW.abs())*torch.eye(model.nu, dtype=dtype)).to(d)
# noise_sigma = 0.2*torch.eye(model.nu, dtype=dtype).to(d)
lambda_ = 0.0025

def dynamics(state, perturbed_action):
    #load state into a MjData instance of the model
    control = perturbed_action.unsqueeze(1).repeat_interleave(10,dim=1).cpu().numpy()
    state, _ = mujoco.rollout.rollout(model=sim_model, data=data_list, initial_state = state.cpu().numpy(), nstep = 10, control=control, persistent_pool=False)
    return torch.tensor(state[:,-1,:]).squeeze(1) # remove the nsteps dimension since it's only simulated for 1 time step

goal_pos = np.array([0.0,0.03,0.1])
goal_quat = np.array([1.0,0.0,0.0,0.0]) #default goal_quat
w_pos = 100
w_rot = 0.1

def running_cost(state,action):
    cost = 0

    qo_pos_traj = state[..., 1:4]
    qo_quat_traj = state[..., 4:8]
    qo_pos_diff = qo_pos_traj - goal_pos
    qo_quat_diff = quat_diff_so3(qo_quat_traj, goal_quat)

    pos_cost = w_pos * 0.5 * np.square(qo_pos_diff).sum(-1).mean(-1)
    rot_cost = w_rot * 0.5 * np.square(qo_quat_diff).sum(-1).mean(-1)
    cost += pos_cost + rot_cost

    return torch.tensor(1000*cost, dtype=dtype, device=d)

def terminal_cost(state):
    cost = 0
    return cost

mppi = custom_mppi.CUSTOM_MPPI(dynamics, running_cost, nx, noise_sigma, use_mujoco_physics=False, terminal_cost = terminal_cost, num_samples=N_SAMPLES, time_steps=TIMESTEPS, steps_per_stage=20,
                         lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d, dtype=dtype),
                         u_max=torch.tensor(ACTION_HIGH, device=d,dtype=dtype), device=d, U_init=torch.tensor(            [
                0.5, -0.75, 0.75, 0.25,  # index
                0.5, 0.0, 0.75, 0.25,  # middle
                0.5, 0.75, 0.75, 0.25,  # ring
                0.65, 0.9, 0.75, 0.6,  # thumb
            ], dtype=dtype,device=d).unsqueeze(0).repeat(TIMESTEPS,1))
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
        print(action)
        # print(f"time: {time.time() - now}")
 
        data.ctrl[:] = action.cpu().numpy()
        for i in range(100):
            mujoco.mj_step(model,data)
            viewer.sync()

# frames = []

# for _ in tqdm(range(300)):
#     state = np.concatenate([[data.time],data.qpos, data.qvel])
#     state = torch.tensor(state, dtype = dtype, device = d)

#     # now = time.time()
#     action, _, _ = mppi.command(state)
#     # print(f"time: {time.time() - now}")

#     data.ctrl[:] = action.cpu().numpy()
#     # for i in range(30):
#     mujoco.mj_step(model,data)
#     mujoco.mjv_updateScene(model,data,mujoco.MjvOption(),None,cam,mujoco.mjtCatBit.mjCAT_ALL, renderer.scene)
#     # renderer.update_scene(data)
#     pixels = renderer.render()
#     frames.append(pixels)

# imageio.mimsave('simulation.mp4', frames, fps = 100)



