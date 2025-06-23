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

model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_go2/scene.xml")
data = mujoco.MjData(model)

TIMESTEPS = 60  # T
N_SAMPLES = 10000  # K
ACTION_LOW = -40.0
ACTION_HIGH = 40.0


d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if d == torch.device("cpu"):
    d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
if d == torch.device("cpu"):
    warnings.warn("No GPU device detected, using cpu instead", UserWarning)
dtype = torch.float32

# noise_sigma = torch.tensor(10, device=d, dtype=dtype)
# noise_sigma = torch.tensor([[2, 0], [0, 2]], device=d, dtype=dtype)

noise_sigma = 20*torch.eye(model.nu, device=d, dtype=dtype)
lambda_ = 1.

def dynamics(state, perturbed_action):
    K = perturbed_action.shape[0]
    state, _ = mujoco.rollout(model=model, data='''TODO''', nstep = 1, control=perturbed_action)
    return state.unsqueeze(1)


def running_cost(state, action):
    cost = 0

    return cost 

def terminal_cost(state):
    cost = 0
    return cost

# mppi = base_mppi.BASE_MPPI(dynamics, running_cost, nx, noise_sigma, terminal_cost = terminal_cost, num_samples=N_SAMPLES, time_steps=TIMESTEPS, steps_per_stage=10,
#                          lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
#                          u_max=torch.tensor(ACTION_HIGH, device=d), device=d)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # state = data.state()
        # action = mppi.command(state)
        action = np.random.uniform(-20,20,size=model.nu)
        data.ctrl[:] = action
        mujoco.mj_step(model,data)
        viewer.sync()
        # mujoco.rollout()


