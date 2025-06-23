import mujoco
import mujoco.rollout
import mujoco.viewer
from mujoco import rollout
import numpy as np
import torch
import warnings

import sys
sys.path.append("..")
import custom_mppi
import base_mppi
import time

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/scene.xml")
data = mujoco.MjData(model)

model.opt.timestep = 0.01

body_name = "go2"
id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

goal = [-0.1934, -0.142, 0.232]

TIMESTEPS = 30  # T
N_SAMPLES = 4000  # K
ACTION_LOW = -40.0
ACTION_HIGH = 40.0

NUM_THREAD = 32

nx = model.nq + model.nv + 1 # +1 since time is considered a state my mujoco


d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if d == torch.device("cpu"):
    d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
d = torch.device("cpu")
if d == torch.device("cpu"):
    warnings.warn("No GPU device detected, using cpu instead", UserWarning)
dtype = torch.float32

# noise_sigma = torch.tensor(10, device=d, dtype=dtype)
# noise_sigma = torch.tensor([[2, 0], [0, 2]], device=d, dtype=dtype)

noise_sigma = 20*torch.eye(model.nu, device=d, dtype=dtype)
lambda_ = 1.

def dynamics(state, perturbed_action, model, data):
    #load state into a MjData instance of the model
    control = perturbed_action.unsqueeze(1).cpu().numpy()
    state, _ = mujoco.rollout.rollout(model=model, data=data, initial_state = state.cpu().numpy(), nstep = 1, control=control, persistent_pool=True)
    return torch.tensor(state).squeeze(1) # remove the nsteps dimension since it's only simulated for 1 time step


def running_cost(state, action):
    cost = 0

    goal = [[ 0.     , 0.  ,    0.    ],
            [ 0.     , 0.      ,0.445 ],
            [ 0.1934,  0.0465  ,0.445 ],
            [ 0.1934 , 0.142   ,0.445 ],
            [ 0.1934 , 0.142   ,0.232 ],
            [ 0.1934 ,-0.0465  ,0.445 ],
            [ 0.1934 ,-0.142   ,0.445 ],
            [ 0.1934 ,-0.142   ,0.232 ],
            [-0.1934 , 0.0465  ,0.445 ],
            [-0.1934 , 0.142   ,0.445 ],
            [-0.1934 , 0.142   ,0.232 ],
            [-0.1934 ,-0.0465  ,0.445 ],
            [-0.1934 ,-0.142   ,0.445 ],
            [-0.1934 ,-0.142   ,0.232 ]]
    goal = np.array(goal)
    model=mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/scene.xml")
    xpos_array = np.zeros((state.shape[0],model.nbody,3))
    data = mujoco.MjData(model)
    for i in range(state.shape[0]):
        data.qpos[:] = state[i,1:1+model.nq]
        data.qvel[:] = state[i, 1+model.nq:]
        mujoco.mj_forward(model,data)
        xpos_array[i] = data.xpos
    cost = cost + np.sum((xpos_array - goal)**2, axis = (1,2))
    return torch.tensor(cost, dtype = torch.float32) 

def terminal_cost(state):
    cost = 0
    return cost

mppi = custom_mppi.CUSTOM_MPPI(dynamics, running_cost, nx, noise_sigma, use_mujoco_dynamics= True, model = model,terminal_cost = terminal_cost, num_samples=N_SAMPLES, time_steps=TIMESTEPS, steps_per_stage=10,
                         lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
                         u_max=torch.tensor(ACTION_HIGH, device=d), device=d)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # vectorize robot states
        state = np.concatenate([[data.time], data.qpos, data.qvel])
        state = torch.tensor(state, dtype = dtype, device = d)
        now = time.time()
        # print(data.xpos)
        action, _, _ = mppi.command(state)
        print(f"time: {time.time() - now}")
        # action = np.random.uniform(-20,20,size=model.nu)
 
        for i in range(5):
            data.ctrl[:] = action.cpu().numpy()[i]
            mujoco.mj_step(model,data)
        viewer.sync()


