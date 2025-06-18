import gymnasium as gym
import numpy as np
import warnings
import torch
import time
import math
from pytorch_mppi import mppi

import sys
sys.path.append("..")
import custom_mppi
import base_mppi

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1" #source code at /opt/anaconda3/envs/mppi_research/lib/python3.10/site-packages/gymnasium/envs/classic_control/pendulum.py
    TIMESTEPS = 20  # T
    N_SAMPLES = 1000 # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0

    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if d == torch.device("cpu"):
        # d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if d == torch.device("cpu"):
        warnings.warn("No GPU device detected, using cpu instead", UserWarning)
    dtype = torch.float64

    noise_sigma = torch.tensor(1, device=d, dtype=dtype)
    # noise_sigma = torch.tensor([[10, 0], [0, 10]], device=d, dtype=dtype)
    lambda_ = 1.


    def dynamics(state, perturbed_action):
        # true dynamics from gym
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = perturbed_action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (3 * g / (2 * l) * torch.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = torch.clip(newthdot, -8, 8)
        newth = th + newthdot * dt

        state = torch.cat((newth, newthdot), dim=1)
        return state


    def angle_normalize(x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)


    def running_cost(state, action):
        theta = state[:, 0]
        theta_dt = state[:, 1]
        action = action[:, 0]
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2
        return cost


    def train(new_data):
        pass


    downward_start = True
    env = gym.make(ENV_NAME, render_mode="human")
    nx = 2

    # env.reset()
    # if downward_start:
    #     env.state = env.unwrapped.state = [np.pi, 1]

    # mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
    #                      lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
    #                      u_max=torch.tensor(ACTION_HIGH, device=d), device=d)
    # start = time.time()
    # total_reward = mppi.run_mppi(mppi_gym, env, train, iter=300)
    # print("Time:", time.time() - start)


    env.reset()
    if downward_start:
        env.state = env.unwrapped.state = [np.pi, 1]

    mppi_gym = base_mppi.BASE_MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, time_steps=TIMESTEPS,
                         lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
                         u_max=torch.tensor(ACTION_HIGH, device=d), device=d)
    start = time.time()    
    total_reward = custom_mppi.run_mppi(mppi_gym, env, iter=200)
    print("Time:", time.time() - start)



    env.reset()
    if downward_start:
        env.state = env.unwrapped.state = [np.pi, 1]

    mppi_gym = custom_mppi.CUSTOM_MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, time_steps=TIMESTEPS, steps_per_stage=5,
                         lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
                         u_max=torch.tensor(ACTION_HIGH, device=d), device=d)
    start = time.time()    
    total_reward = custom_mppi.run_mppi(mppi_gym, env, iter=200)
    print("Time:", time.time() - start)

    # logger.info("Total reward %f", total_reward)

    env.close()