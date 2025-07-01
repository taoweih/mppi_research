import gymnasium as gym
import pygame
import warnings
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_mppi import mppi
import custom_envs
import sys
sys.path.append("..")
import custom_mppi
import base_mppi

if __name__ == "__main__":
    ENV_NAME = "ObstacleAvoidance-v0"

    TIMESTEPS = 60  # T
    N_Chunks = 5
    N_SAMPLES = 3000  # K
    ACTION_LOW = -3.0
    ACTION_HIGH = 3.0
    ENV = "U"
    # ENV = "default"

    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == torch.device("cpu"):
        d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # d = torch.device("cpu")
    if d == torch.device("cpu"):
        warnings.warn("No GPU device detected, using cpu instead", UserWarning)
    dtype = torch.float32

    # noise_sigma = torch.tensor(10, device=d, dtype=dtype)
    noise_sigma = torch.tensor([[2, 0], [0, 2]], device=d, dtype=dtype)
    lambda_ = 1.

    if ENV == "default":
        start_position = [50,50, 0, 0]
        goal = torch.tensor([550.0, 350.0], device=d, dtype=dtype)
        obstacles = [
                pygame.Rect(200, 100, 100, 300),
                pygame.Rect(400, 200, 100, 50),
            ]
    elif ENV == "U":
        start_position = [330,200, 0, 0]
        goal = torch.tensor([550.0, 200.0], device=d, dtype=dtype)
        obstacles = [
                pygame.Rect(200, 150, 200, 10),
                pygame.Rect(200, 250, 200, 10),
                pygame.Rect(390, 160, 10,  90),
            ]
    obs_map = torch.zeros((600,400))
    for obs in obstacles:
        obs_map[obs.left:obs.left + obs.width,obs.top:obs.top + obs.height] = 1

    max_speed = 8
    max_acceleartion = 3

    def dynamics(state, perturbed_action):

        car_x = state[:, 0]
        car_y = state[:, 1]
        speed_x = state[:, 2]
        speed_y = state[:, 3]

        u = torch.clamp(perturbed_action, -1*max_acceleartion, max_acceleartion)  # shape (k,2)
        speed_x = torch.clamp(speed_x + u[:,0], -1*max_speed, max_speed)
        speed_y = torch.clamp(speed_y + u[:,1], -1*max_speed, max_speed)
        new_x = car_x + speed_x
        new_y = car_y + speed_y

        valid_x = (new_x >= 0) & (new_x < 599)
        valid_y = (new_y >= 0) & (new_y < 399)

        int_x = new_x.long().clamp(0, obs_map.shape[0] - 1).cpu()
        int_y = new_y.long().clamp(0, obs_map.shape[1] - 1).cpu()

        is_obstacle = obs_map[int_x, int_y] == 1
        is_obstacle = is_obstacle.to(valid_x.device)
        valid_mask = valid_x & valid_y & (~is_obstacle)

        new_x = torch.where(valid_mask, new_x, car_x)
        new_y = torch.where(valid_mask, new_y, car_y)

        state = torch.stack([new_x, new_y, speed_x,speed_y], dim=1)
        return state


    def running_cost(state, action):
        cost = 0
        distance_sq = torch.sum((state[:,0:2] - goal)**2, dim=1)
        cost += distance_sq

        for obs in obstacles:
            x_in = (state[:, 0] >= obs.left) & (state[:, 0] <= obs.left + obs.width)
            y_in = (state[:, 1] >= obs.top) & (state[:, 1] <= obs.top + obs.height)
            collided = x_in & y_in

            cost += collided.float() *1e18  # penalty
            # cost += 0.1*torch.sum(state[:,2:]**2, dim=1)

        return cost 
    
    def terminal_cost(state):
        cost = torch.sum((state[:,0:2] - goal)**8, dim=1)
        return cost


    def train(new_data):
        pass



    env = gym.make(ENV_NAME, render_mode="human", env = ENV)
    nx = 2


    
    # for _ in range(1000):
    #     action = np.array([4,0])
    #     _,r,_,_,_ = env.step(action)


    # env.reset()
    # env.state = env.unwrapped.state = start_position

    # mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
    #                      lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
    #                      u_max=torch.tensor(ACTION_HIGH, device=d), device=d)
    # total_reward = mppi.run_mppi(mppi_gym, env, train, iter=200)

    # env.reset()
    # env.state = env.unwrapped.state = start_position

    # mppi_gym = base_mppi.BASE_MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, time_steps=TIMESTEPS,
    #                      lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
    #                      u_max=torch.tensor(ACTION_HIGH, device=d), device=d)
    # total_reward = base_mppi.run_mppi(mppi_gym, env, iter=100)

    s = np.zeros(10)
    for j in tqdm(range(10,101,10)):
        TIMESTEPS = j
        for _ in tqdm(range(100)):
            env.reset()
            env.state = env.unwrapped.state = start_position

            mppi_gym = custom_mppi.CUSTOM_MPPI(dynamics, running_cost, nx, noise_sigma, terminal_cost = terminal_cost, num_samples=N_SAMPLES, time_steps=TIMESTEPS, steps_per_stage = int(TIMESTEPS/N_Chunks),
                                lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
                                u_max=torch.tensor(ACTION_HIGH, device=d), device=d)

            for i in range(200):
                state = env.unwrapped.state.copy() # current state of the robot
                e = np.sum(np.sqrt((np.array(state[0:2]) - goal.cpu().numpy())**2))
                if e < 100:
                    s[int(j/10)-1] +=1
                    break

                action, states, policy = mppi_gym.command(state) # get the control input from mppi based on current state

                _ = env.step(action.cpu().numpy()) # execute the control input (env return info for RL, can be discarded)
                # env.unwrapped.set_render_info(states.cpu().numpy(), policy.cpu().numpy())
                # env.render()

            env.close()
    print(s)
    plt.figure()
    plt.plot(range(10,101,10),s)
    plt.xlabel("Horizon (TIMESTEPS)")
    plt.ylabel("Success rate %")
    plt.show()


