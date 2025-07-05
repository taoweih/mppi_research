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
MODEL = mujoco.MjModel.from_xml_string(model_xml)

model = MODEL
data = mujoco.MjData(model)
model.opt.timestep = 0.01
# mjx_model = mjx.put_model(model)

# body_name = "go2"
# id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

TIMESTEPS = 10  # T
N_SAMPLES = 1000  # K
ACTION_LOW = -5.0
ACTION_HIGH = 5.0

nx = model.nq + model.nv # +1 since time is considered a state my mujoco


d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if d == torch.device("cpu"):
    d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if d == torch.device("cpu"):
    warnings.warn("No GPU device detected, using cpu instead", UserWarning)
dtype = torch.float32

# noise_sigma = torch.tensor(10, device=d, dtype=dtype)
# noise_sigma = torch.tensor([[2, 0], [0, 2]], device=d, dtype=dtype)

noise_sigma = 2*torch.eye(model.nu, device=d, dtype=dtype)
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

def dynamics(state, perturbed_action, model, data):
    #load state into a MjData instance of the model
    control = perturbed_action.unsqueeze(1).cpu().numpy()
    state, _ = mujoco.rollout.rollout(model=model, data=data, initial_state = state.cpu().numpy(), nstep = 1, control=control, persistent_pool=True)
    return torch.tensor(state).squeeze(1) # remove the nsteps dimension since it's only simulated for 1 time step


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
#     # goal_quat = jnp.array( [[1.0, 0.0, 0, 0,],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.],
#     #             [1, 0, 0, 0.]])


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

# mppi = custom_mppi.CUSTOM_MPPI(dynamics, running_cost, nx, noise_sigma, use_mujoco_physics=True, terminal_cost = terminal_cost, num_samples=N_SAMPLES, time_steps=TIMESTEPS, steps_per_stage=10,
#                          lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
#                          u_max=torch.tensor(ACTION_HIGH, device=d), device=d)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # # vectorize robot states
        state = np.concatenate([data.qpos, data.qvel])
        # print(f'xpos:{jnp.array(data.xpos)}')
        # print(f'xquad:{jnp.array(data.xquat)}')
        state = torch.tensor(state, dtype = dtype, device = d)
        # now = time.time()
        # action, _, _ = mppi.command(state)
        # print(f"time: {time.time() - now}")
        action = torch.zeros_like(torch.tensor(np.random.uniform(-5,5,size=model.nu)))
 
        # for i in range(5):
        data.ctrl[:] = action.cpu().numpy()
        print(action)
        mujoco.mj_step(model,data)
        viewer.sync()


