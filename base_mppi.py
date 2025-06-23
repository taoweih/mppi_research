import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from arm_pytorch_utilities import handle_batch_input
from tqdm import tqdm
# Built based on the base MPPI implementation from pytorch_mppi form https://github.com/UM-ARM-Lab/pytorch_mppi/blob/master/src/pytorch_mppi/mppi.py

class BASE_MPPI():

    def __init__(self, dynamics, running_cost, nx, noise_sigma, 
                 terminal_cost = None,
                 noise_mu=None,
                 num_samples=100, 
                 time_steps=15, 
                 device = torch.device("cpu"),
                 lambda_ = 1.0,
                 u_min=None,
                 u_max=None,
                 u_init=None,
                 U_init = None):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param num_samples: K, number of trajectories to sample
        :param time_steps: T, length of each trajectory
        :param device: pytorch device
        :param lambda\_: temperature, positive scalar where larger values will allow more exploration
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        """
        
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
        self.noise_mu = noise_mu.to(self.device) # mean
        self.noise_sigma = noise_sigma.to(self.device) # variance
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu,covariance_matrix=self.noise_sigma) # distribution to sample control input from
        # self.noise_dist = torch.distributions.uniform.Uniform(self.u_min*torch.ones(self.nu, dtype=self.dtype), self.u_max*torch.ones(self.nu, dtype=self.dtype))

        # Control sequence: (T x nu)
        self.U = U_init
        self.u_init = u_init.to(self.device)
        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.F = dynamics
        self.running_cost = running_cost
        self.state = None
        self.terminal_cost = terminal_cost
    
    def reset(self):
        self.U = self.noise_dist.sample((self.T,))

    #handle parallel dynamics and running cost function
    @handle_batch_input(n=1)
    def _dynamics(self,state,u,t):
        return self.F(state,u)
    
    @handle_batch_input(n=1)
    def _running_cost(self,state,u,t):
        return self.running_cost(state,u)
    
    def shift_nominal_trajectory(self): 
        '''
        Shift the control sequence (Self.U) after applying the first control input
        '''
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

    def get_perturbed_actions_and_noise(self):
        '''
        Draw K samples of random control actions in T timesteps.

        :returns perturbed_action: Shape (K x T x nu).
        :returns noise: Shape (K x T x nu).
        '''
        noise = self.noise_dist.rsample((self.K, self.T))
        perturbed_action = self.U + noise
        perturbed_action = torch.max(torch.min(perturbed_action, self.u_max), self.u_min) # control limit
        noise = perturbed_action - self.U # update noise since perturbed action is changed
        return perturbed_action, noise # (K x T x nu)
    
    def implement_rollouts(self, perturbed_actions):
        '''
        Rollout the K samples of control input sequence and evaluate the cost

        :params perturbed_actions: Shape (K x T x nu).

        :returns rollout_cost: Shape (K).
        '''
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu
        rollout_cost = torch.zeros(K, device=self.device, dtype=self.dtype)

        state = self.state.view(1, -1).repeat(K, 1) # repeat K starting state

        # rollout dynamics and calculate cost for T timesteps
        for t in range(T):
            u = perturbed_actions[:,t]
            next_state = self._dynamics(state, u, t)
            state = next_state
            running_cost = self._running_cost(state,u,t)
            rollout_cost = rollout_cost + running_cost

        return rollout_cost
    
    def compute_optimal_control_sequence(self,cost_total, noise):
        '''
        Based on the cost, weight each sample and combine them to find a optimal control trajectory

        :params cost_total: Shape (K).
        :params nosie: Shape (K x T x nu)

        :returns action: Shape (T x nu).
        '''
        weight = torch.exp((-1/self.lambda_)*(cost_total-torch.min(cost_total))) # subtract min to prevent extreme small weight)
        omega = weight/torch.sum(weight)
        action = torch.sum(omega.view(-1,1,1)*noise,dim=0)
        return action
        
    def command(self,state):
        '''
        Main step for MPPI algorithm, shift control input, generate K samples of control input sequence, 
        rollout and combine to find one optimal control input

        :params state: Current state of the system. Shape (nx)

        :returns control_input: Shape (nu).
        '''
        self.shift_nominal_trajectory()

        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype,device=self.device)

        cost_total = torch.zeros(self.K, device=self.device, dtype=self.dtype)

        # Generate samples of control input sequences and get rollout cost
        perturbed_actions, noise = self.get_perturbed_actions_and_noise()
        rollout_cost = self.implement_rollouts(perturbed_actions)

        # Control cost
        action_cost = self.lambda_ * noise @ self.noise_sigma_inv
        control_cost = torch.sum(self.U * action_cost, dim=(1, 2))

        # Combine all the cost
        cost_total = cost_total + control_cost + rollout_cost

        # Combine to find one optimal trajectory based on total cost
        action = self.compute_optimal_control_sequence(cost_total, noise)
        self.U = self.U + action
        control_input = self.U[0]

        return control_input      

def run_mppi(mppi, env, iter=100, render = True):
    '''
    Run mppi algorithm for a set of iterations 

    :params mppi: an instance of MPPI class
    :params env: a environment
    :params iter: number of iterations to run mppi
    :params render: whether to display the gym environment
    '''
    
    for i in tqdm(range(iter)):
        state = env.unwrapped.state.copy() # current state of the robot
        action = mppi.command(state) # get the control input from mppi based on current state
        _ = env.step(action.cpu().numpy()) # execute the control input (env return info for RL, can be discarded)
        if render:
            env.render()
    return