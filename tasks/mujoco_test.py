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
model_xml = r"""<mujoco model="cylinder_push">
  <option timestep="0.02" />

  <asset>
    <texture name="blue_grid" type="2d" builtin="checker" rgb1=".02 .14 .44" rgb2=".27 .55 1" width="300" height="300" mark="edge" markrgb="1 1 1"/>
    <material name="blue_grid" texture="blue_grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <default>
    <default class="slider">
      <position kp="10" ctrlrange="-10 10" forcerange="-1000 1000"/>
    </default>
  </default>

  <worldbody>
    <body>
      <geom mass="0" name="floor" pos="0 0 -0.25" condim="3" size="10.0 10.0 0.10" rgba="0 1 1 1" type="box" material="blue_grid"/>
    </body>

    <body name="pusher" pos="0 0 0">
      <joint name="slider_x" damping="4" type="slide" axis="1 0 0" />
      <joint name="slider_y" damping="4" type="slide" axis="0 1 0" />
      <geom name="pusher" type="cylinder" size="0.25 0.1" mass="1" rgba=".9 .5 .5 1" friction="0"/>
      <site pos="0 0 0.15" name="pusher_site"/>
    </body>

    <body name="cart" pos="0 0 0">
      <joint name="slider_cart_x" damping="4" type="slide" axis="1 0 0" />
      <joint name="slider_cart_y" damping="4" type="slide" axis="0 1 0" />
      <geom name="cart" type="cylinder" size="0.25 0.1" mass="1" rgba=".1 .5 .5 1" friction="0"/>
      <site pos="0 0 0.15" name="cart_site"/>
    </body>
  </worldbody>

  <actuator>
    <position name="actuator_pusher_x" joint="slider_x" class="slider" />
    <position name="actuator_pusher_y" joint="slider_y" class="slider" />
  </actuator>

  <sensor>
    <framepos name="trace_pusher" objtype="site" objname="pusher_site"/>
    <framepos name="trace_cart" objtype="site" objname="cart_site"/>
  </sensor>

</mujoco>"""


# MODEL = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/scene_mjx.xml")

NUM_THREAD = 16
MODEL = mujoco.MjModel.from_xml_string(model_xml)

model = MODEL
data = mujoco.MjData(model)
model.opt.timestep = 0.01

data_list = [mujoco.MjData(model) for _ in range(NUM_THREAD)]
# mjx_model = mjx.put_model(model)

TIMESTEPS = 100  # T
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

# noise_sigma = torch.tensor(10, device=d, dtype=dtype)
# noise_sigma = torch.tensor([[2, 0], [0, 2]], device=d, dtype=dtype)
noise_sigma = 0.1*torch.eye(model.nu, device=d, dtype=dtype)
lambda_ = 1.

# for compatible mjx model
# @jax.jit
# @jax.vmap
# def dynamics(state, perturbed_action):
#     # mjx_model = mjx.put_model(model)
#     mjx_data = mjx.make_data(mjx_model)

#     qpos = state[:model.nq]
#     qvel = state[model.nq:]

#     mjx_data= mjx_data.replace(qpos=qpos, qvel=qvel,ctrl=perturbed_action)
#     mjx_data_next = mjx.step(mjx_model,mjx_data)

#     state = jnp.concatenate([mjx_data_next.qpos, mjx_data_next.qvel])

#     return state

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

    return torch.tensor(10000*cost, dtype=dtype, device=d)


# # for mjx compatible models
# @jax.jit
# @jax.vmap
# def running_cost(state, action):
#     cost = 0
#     # goal_pos = jnp.array([[ 0,      0,      0    ],
#     #                     [ 0.     , 0.     , 0.445 ],
#     #                     [ 0.1934 , 0.0465  ,0.445 ],
#     #                     [ 0.1934 , 0.142   ,0.445 ],
#     #                     [ 0.1934 , 0.142   ,0.232 ],
#     #                     [ 0.1934 ,-0.0465  ,0.445 ],
#     #                     [ 0.1934 ,-0.142   ,0.445 ],
#     #                     [ 0.1934 ,-0.142   ,0.232 ],
#     #                     [-0.1934 , 0.0465  ,0.445 ],
#     #                     [-0.1934 , 0.142   ,0.445 ],
#     #                     [-0.1934 , 0.142   ,0.232 ],
#     #                     [-0.1934 ,-0.0465  ,0.445 ],
#     #                     [-0.1934 ,-0.142   ,0.445 ],
#     #                     [-0.1934 ,-0.142   ,0.232 ]])

#     # mjx_model = mjx.put_model(model)
#     # mjx_data = mjx.make_data(mjx_model)

#     # qpos = state[:model.nq]
#     # qvel = state[model.nq:]

#     # mjx_data= mjx_data.replace(qpos=qpos, qvel=qvel,ctrl=action)
#     # next_data = mjx.forward(mjx_model, mjx_data)
#     # curr_xpos = next_data.xpos
#     # curr_xquat = next_data.xquat

#     # cost = cost + jnp.sum((curr_xpos - goal_pos)**2) + jnp.sum((curr_xquat - goal_quat)**2)

#     return cost

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
        # print(action)
        # print(f"time: {time.time() - now}")
 
        data.ctrl[:] = action.cpu().numpy()
        mujoco.mj_step(model,data)
        viewer.sync()


