import gymnasium as gym
import pygame
import warnings
import torch
import time
from pytorch_mppi import mppi
import custom_envs
import sys
sys.path.append("..")
import custom_mppi

if __name__ == "__main__":
    ENV_NAME = "ObstacleAvoidance-v0"

    TIMESTEPS = 200  # T
    N_SAMPLES = 5000  # K
    ACTION_LOW = -5.0
    ACTION_HIGH = 5.0
    # ENV = "U"
    ENV = "default"

    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == torch.device("cpu"):
        d = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    d = torch.device("cpu")
    if d == torch.device("cpu"):
        warnings.warn("No GPU device detected, using cpu instead", UserWarning)
    dtype = torch.float32

    # noise_sigma = torch.tensor(10, device=d, dtype=dtype)
    noise_sigma = torch.tensor([[3, 0], [0, 3]], device=d, dtype=dtype)
    lambda_ = 1.

    if ENV == "default":
        start_position = [50,50]
        goal = torch.tensor([550.0, 350.0], device=d, dtype=dtype)
        obstacles = [
                pygame.Rect(200, 100, 100, 300),
                pygame.Rect(400, 200, 100, 50),
            ]
    elif ENV == "U":
        start_position = [300,200]
        goal = torch.tensor([550.0, 350.0], device=d, dtype=dtype)
        obstacles = [
                pygame.Rect(200, 100, 200, 10),
                pygame.Rect(200, 300, 200, 10),
                pygame.Rect(390, 110, 10, 190),
            ]
    obs_map = torch.zeros((500,200))
    for obs in obstacles:
        obs_map[obs.left:obs.left + obs.width,obs.top:obs.top + obs.height] = 1


    def dynamics(state, perturbed_action):

        car_x = state[:, 0]
        car_y = state[:, 1]

        u = torch.clamp(perturbed_action, -10.0, 10.0)  # shape (k,2)
        dx = u[:, 0]
        dy = u[:, 1]

        new_x = car_x + dx
        new_y = car_y + dy

        valid_x = (new_x >= 0) & (new_x < 599)
        valid_y = (new_y >= 0) & (new_y < 399)

        int_x = new_x.long().clamp(0, obs_map.shape[0] - 1).cpu()
        int_y = new_y.long().clamp(0, obs_map.shape[1] - 1).cpu()

        is_obstacle = obs_map[int_x, int_y] == 1
        is_obstacle = is_obstacle.to(valid_x.device)
        valid_mask = valid_x & valid_y & (~is_obstacle)

        new_x = torch.where(valid_mask, new_x, car_x)
        new_y = torch.where(valid_mask, new_y, car_y)

        state = torch.stack([new_x, new_y], dim=1)
        return state


    def running_cost(state, action):
        cost = 0
        distance_sq = torch.sum((state - goal)**4, dim=1)
        cost += distance_sq

        for obs in obstacles:
            x_in = (state[:, 0] >= obs.left) & (state[:, 0] <= obs.left + obs.width)
            y_in = (state[:, 1] >= obs.top) & (state[:, 1] <= obs.top + obs.height)
            collided = x_in & y_in

            cost += collided.float() *1e15  # penalty

        return cost 


    def train(new_data):
        pass



    env = gym.make(ENV_NAME, render_mode="human", env = ENV)
    nx = 2

    env.reset()
    env.state = env.unwrapped.state = start_position

    
    # for _ in range(1000):
    #     action = np.array([4,0])
    #     _,r,_,_,_ = env.step(action)


    mppi_gym = mppi.MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                         lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
                         u_max=torch.tensor(ACTION_HIGH, device=d), device=d)
    start = time.time()
    total_reward = mppi.run_mppi(mppi_gym, env, train, iter=100)
    print("Time:", time.time() - start)


    env.reset()
    env.state = env.unwrapped.state = start_position

    mppi_gym = custom_mppi.CUSTOM_MPPI(dynamics, running_cost, nx, noise_sigma, num_samples=N_SAMPLES, time_steps=TIMESTEPS,
                         lambda_=lambda_, u_min=torch.tensor(ACTION_LOW, device=d),
                         u_max=torch.tensor(ACTION_HIGH, device=d), device=d)
    start = time.time()
    total_reward = custom_mppi.run_mppi(mppi_gym, env, iter=100)
    print("Time:", time.time() - start)

    env.close()