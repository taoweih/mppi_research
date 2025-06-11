import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from arm_pytorch_utilities import handle_batch_input

class CUSTOM_MPPI():

    def __init__(self, dynamics, running_cost, nx, noise_sigma, 
                 noise_mu=None,
                 terminal_cost = None, 
                 num_samples=100, 
                 time_steps=15, 
                 device = torch.device("cpu"),
                 lambda_ = 1.0,
                 u_min=None,
                 u_max=None,
                 u_init=None,
                 U_init = None):
        
        self.device = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples
        self.T = time_steps

        # dimensions of state and control
        self.nx = nx # state dimension
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0] # control input dimension
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)
        
        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # 1D edge case:
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1,1)

        # control limit:
        self.u_min = u_min
        self.u_max = u_max

        if not torch.is_tensor(self.u_min):
            self.u_min = torch.tensor(u_min)
        if not torch.is_tensor(self.u_max):
            self.u_max = torch.tensor(u_max)

        self.u_min = self.u_min.to(device=self.device)
        self.u_max = self.u_max.to(device=self.device)

        # added rollout control noise
        self.noise_mu = noise_mu.to(self.device)
        self.noise_sigma = noise_sigma.to(self.device)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu,covariance_matrix=self.noise_sigma)

        # Control sequence: (T x nu)
        self.U = U_init
        self.u_init = u_init.to(self.device)
        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))
            # self.U = torch.zeros_like(self.U)

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost 
        self.state = None

    @handle_batch_input(n=1)
    def _dynamics(self,state,u,t):
        return self.F(state,u)
    
    @handle_batch_input(n=1)
    def _running_cost(self,state,u,t):
        return self.running_cost(state,u)
    
    def shift_nominal_trajectory(self):
        self.U = torch.roll(self.U, -1, dims=0)
        # self.U[-1] = self.u_init
        self.U[-1] = self.U[-2]

    def reset(self):
        self.U = self.noise_dist.sample((self.T,))
        # self.U = torch.zeros_like(self.U)


    def get_perturbed_actions_and_noise(self):
        noise = self.noise_dist.rsample((self.K, self.T))
        perturbed_action = self.U + noise
        perturbed_action = torch.max(torch.min(perturbed_action, self.u_max), self.u_min) # control limit
        return perturbed_action, noise # (K x T x nu)
    
    def implement_rollouts(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu
        total_cost = torch.zeros(K, device=self.device, dtype=self.dtype)

        state = self.state.view(1, -1).repeat(K, 1)
        states = []
        actions = []

        for t in range(T):
            u = perturbed_actions[:,t]
            next_state = self._dynamics(state, u, t)
            state = next_state
            running_cost = self._running_cost(state,u,t)
            total_cost = total_cost + running_cost

            states.append(state)
            actions.append(u)

        actions = torch.stack(actions, dim=-2) # size K x T x nu
        states = torch.stack(states, dim=-2) # size K x T x nx


        return total_cost
    
    def compute_optimal_control_sequence(self,cost_total, noise):
        weight = torch.exp((-1/self.lambda_)*(cost_total-torch.min(cost_total))) # subtract min to prevent extreme small weight)
        omega = weight/torch.sum(weight)
        action = torch.sum(omega.view(-1,1,1)*noise,dim=0)
        return action
        
    def command(self,state):
        self.shift_nominal_trajectory
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype,device=self.device)

        perturbed_actions, noise = self.get_perturbed_actions_and_noise()
        cost_total = self.implement_rollouts(perturbed_actions)

        action_cost = self.lambda_ * torch.abs(noise) @ self.noise_sigma_inv
        control_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        cost_total = cost_total + 0.01*control_cost

        action = self.compute_optimal_control_sequence(cost_total, noise)
        self.U = self.U + action
        return self.U[0]      

def run_mppi(mppi, env, iter=100, render = True):
    for i in range(iter):
        state = env.unwrapped.state.copy() # current state of the robot
        action = mppi.command(state) # get the control input from mppi based on current state
        _ = env.step(action.cpu().numpy()) # execute the control input (env return info for RL, can be discarded)
        if render:
            env.render()
    return