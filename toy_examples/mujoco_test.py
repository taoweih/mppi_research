import mujoco
import mujoco.rollout
import mujoco.viewer

import jax
import jax.numpy as jnp
from mujoco import mjx

import numpy as np
import torch
import warnings

import sys
sys.path.append("..")
import custom_mppi
import base_mppi
import time

MODEL = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/scene_mjx.xml")

model = MODEL
data = mujoco.MjData(model)
# model.opt.timestep = 0.01
mjx_model = mjx.put_model(model)

body_name = "go2"
id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

goal = [-0.1934, -0.142, 0.232]

TIMESTEPS = 20  # T
N_SAMPLES = 4000  # K
ACTION_LOW = -20.0
ACTION_HIGH = 20.0

nx = model.nq + model.nv # +1 since time is considered a state my mujoco


d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if d == torch.device("cpu"):
    d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if d == torch.device("cpu"):
    warnings.warn("No GPU device detected, using cpu instead", UserWarning)
dtype = torch.float32

# noise_sigma = torch.tensor(10, device=d, dtype=dtype)
# noise_sigma = torch.tensor([[2, 0], [0, 2]], device=d, dtype=dtype)

noise_sigma = 10*torch.eye(model.nu, device=d, dtype=dtype)
lambda_ = 1.

@jax.jit
@jax.vmap
def dynamics(state, perturbed_action):
    # mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(mjx_model)

    qpos = state[:model.nq]
    qvel = state[model.nq:]

    mjx_data= mjx_data.replace(qpos=qpos, qvel=qvel,ctrl=perturbed_action)
    mjx_data_next = mjx.step(mjx_model,mjx_data)

    state = jnp.concatenate([mjx_data_next.qpos, mjx_data_next.qvel])

    return state

@jax.jit
@jax.vmap
def running_cost(state, action):
    cost = 0
    # goal = jnp.array([[ 0,      0,      0    ],
    #                     [ 0.     , 0.     , 0.445 ],
    #                     [ 0.1934 , 0.0465  ,0.445 ],
    #                     [ 0.1934 , 0.142   ,0.445 ],
    #                     [ 0.1934 , 0.142   ,0.232 ],
    #                     [ 0.1934 ,-0.0465  ,0.445 ],
    #                     [ 0.1934 ,-0.142   ,0.445 ],
    #                     [ 0.1934 ,-0.142   ,0.232 ],
    #                     [-0.1934 , 0.0465  ,0.445 ],
    #                     [-0.1934 , 0.142   ,0.445 ],
    #                     [-0.1934 , 0.142   ,0.232 ],
    #                     [-0.1934 ,-0.0465  ,0.445 ],
    #                     [-0.1934 ,-0.142   ,0.445 ],
    #                     [-0.1934 ,-0.142   ,0.232 ]])


    # mjx_model = mjx.put_model(model)
    # mjx_data = mjx.make_data(mjx_model)

    # qpos = state[:model.nq]
    # qvel = state[model.nq:]

    # mjx_data= mjx_data.replace(qpos=qpos, qvel=qvel,ctrl=action)
    # next_data = mjx.forward(mjx_model, mjx_data)
    # curr_xpos = next_data.xpos

    # cost = cost + jnp.sum((curr_xpos - goal)**2)

    return cost

def terminal_cost(state):
    cost = 0
    return cost

mppi = custom_mppi.CUSTOM_MPPI(dynamics, running_cost, nx, noise_sigma, use_mujoco_physics=True, terminal_cost = terminal_cost, num_samples=N_SAMPLES, time_steps=TIMESTEPS, steps_per_stage=10,
                         lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
                         u_max=torch.tensor(ACTION_HIGH, device=d), device=d)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # # vectorize robot states
        state = np.concatenate([data.qpos, data.qvel])
        # print(jnp.array(data.xpos))
        state = torch.tensor(state, dtype = dtype, device = d)
        # now = time.time()
        action, _, _ = mppi.command(state)
        # print(f"time: {time.time() - now}")
        # action = torch.zeros_like(torch.tensor(np.random.uniform(-5,5,size=model.nu)))
 
        # for i in range(5):
        data.ctrl[:] = action.cpu().numpy()
        print(action)
        mujoco.mj_step(model,data)
        viewer.sync()


