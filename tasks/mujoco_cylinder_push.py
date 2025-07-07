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

# from judo mpc cylinder push task
ROOT = Path(__file__).resolve().parent
model_xml_path = str(ROOT/"models/judo_mpc_models/xml/cylinder_push.xml")

NUM_THREAD = 16
MODEL = mujoco.MjModel.from_xml_path(model_xml_path)

model = MODEL
data = mujoco.MjData(model)
model.opt.timestep = 0.01

data_list = [mujoco.MjData(model) for _ in range(NUM_THREAD)]

TIMESTEPS = 30  # T
N_SAMPLES = 1000  # K
ACTION_LOW = -10.0
ACTION_HIGH = 10.0

nx = model.nq + model.nv # +1 since time is considered a state my mujoco


d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if d == torch.device("cpu"):
    d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if d == torch.device("cpu"):
    warnings.warn("No GPU device detected, using cpu instead", UserWarning)
dtype = torch.float32

noise_sigma = 2*torch.eye(model.nu, device=d, dtype=dtype)
lambda_ = 1.

def dynamics(state, perturbed_action):
    #load state into a MjData instance of the model
    control = perturbed_action.unsqueeze(1).cpu().numpy()
    state, _ = mujoco.rollout.rollout(model=model, data=data_list, initial_state = state.cpu().numpy(), nstep = 1, control=control, persistent_pool=True)
    return torch.tensor(state).squeeze(1) # remove the nsteps dimension since it's only simulated for 1 time step

goal_pos = [2.0,2.0]
pusher_goal_offset = 0.25
w_pusher_proximity = 0.5
w_pusher_velocity = 0.0
w_cart_position = 0.1

def running_cost(state,action):
    cost = 0
    batch_size = state.shape[0]

    pusher_pos = state[..., 1:3].cpu().numpy()
    cart_pos = state[..., 3:5].cpu().numpy()
    pusher_vel = state[..., 5:7].cpu().numpy()
    cart_goal = np.array(goal_pos)

    cart_to_goal = cart_goal - cart_pos
    cart_to_goal_norm = np.linalg.norm(cart_to_goal, axis=-1, keepdims=True)
    cart_to_goal_direction = cart_to_goal / cart_to_goal_norm

    pusher_goal = cart_pos - pusher_goal_offset * cart_to_goal_direction

    pusher_proximity = 0.5*np.square(pusher_pos - pusher_goal)
    pusher_reward = -w_pusher_proximity * pusher_proximity.sum(-1)

    velocity_reward = -w_pusher_velocity * 0.5*np.square(pusher_vel).sum(-1)

    goal_proximity = 0.5*np.square(cart_pos - cart_goal)
    goal_reward = -w_cart_position * goal_proximity.sum(-1)

    assert pusher_reward.shape == (batch_size,)
    assert velocity_reward.shape == (batch_size,)
    assert goal_reward.shape == (batch_size,)

    cost += -(pusher_reward+velocity_reward + goal_reward)

    return torch.tensor(1*cost, dtype=dtype, device=d)

def terminal_cost(state):
    cost = 0
    return cost

mppi = custom_mppi.CUSTOM_MPPI(dynamics, running_cost, nx, noise_sigma, use_mujoco_physics=False, terminal_cost = terminal_cost, num_samples=N_SAMPLES, time_steps=TIMESTEPS, steps_per_stage=20,
                         lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
                         u_max=torch.tensor(ACTION_HIGH, device=d), device=d)
# initilization
theta = 2*np.pi*np.random.rand(2)
data.qpos = np.array([np.cos(theta[0]),
                     np.sin(theta[0]),
                    2*np.cos(theta[1]),
                    2*np.sin(theta[1])])
data.qvel=np.zeros(4)
mujoco.mj_forward(model,data)


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(viewer.user_scn.geoms[0],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size = [0.1,0,0],
                            pos = [goal_pos[0],goal_pos[1],-0.1],
                            mat=np.eye(3).flatten(),
                            rgba = [1,0,0,1])

        # # vectorize robot states
        state = np.concatenate([[data.time],data.qpos, data.qvel])
        state = torch.tensor(state, dtype = dtype, device = d)

        # now = time.time()
        action, _, _ = mppi.command(state)
        # print(f"time: {time.time() - now}")
 
        data.ctrl[:] = action.cpu().numpy()
        mujoco.mj_step(model,data)
        viewer.sync()


