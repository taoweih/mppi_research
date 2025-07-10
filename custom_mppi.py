import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from arm_pytorch_utilities import handle_batch_input
from torchkde import KernelDensity
import time
from tqdm import tqdm
import jax.numpy as jnp
import jax

# Built based on the base MPPI implementation from pytorch_mppi form https://github.com/UM-ARM-Lab/pytorch_mppi/blob/master/src/pytorch_mppi/mppi.py


class CUSTOM_MPPI():

    def __init__(self, dynamics, running_cost, nx, noise_sigma, 
                 use_mujoco_physics = False,
                 terminal_cost = None,
                 noise_mu=None,
                 num_samples=100, 
                 time_steps=15, 
                 steps_per_stage=5,
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
        :param steps_per_stage, N steps to partially rollout control per stage
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
        self.N = steps_per_stage

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

        # Control sequence: (T x nu)
        self.U = U_init
        self.u_init = u_init.to(self.device)
        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.state = None
        
        self.k_states = None

        self.use_mujoco_physics = use_mujoco_physics
    
    def reset(self):
        self.U = self.noise_dist.sample((self.T,))

    # handle parallel dynamics and running cost function
    # @handle_batch_input(n=1)
    def _dynamics(self,state,u,t):
        if self.use_mujoco_physics:
            state = jnp.array(state.cpu().numpy())
            u = jnp.array(u.cpu().numpy())
            state_next = self.F(state,u)
            return torch.from_numpy(jax.device_get(state_next).copy()).to(self.device)
        else:
            return self.F(state,u)
    
    # @handle_batch_input(n=1)
    def _running_cost(self,state,u,t):
        if self.use_mujoco_physics:
            state = jnp.array(state.cpu().numpy())
            u = jnp.array(u.cpu().numpy())
            cost = self.running_cost(state,u)
            return torch.from_numpy(jax.device_get(cost).copy()).to(self.device)
        else:
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
        # print(f'noise{noise[0]}')
        perturbed_action = self.U + noise
        perturbed_action = torch.max(torch.min(perturbed_action, self.u_max), self.u_min) # control limit
        # print(f'action{perturbed_action[0]}')
        noise = perturbed_action - self.U # update noise since perturbed action is changed
        return perturbed_action, noise # (K x T x nu)
    
    def implement_rollouts(self, perturbed_actions):
        '''
        Rollout the K samples of control input sequence and evaluate the cost

        :params perturbed_actions: Shape (K x T x nu).

        :returns rollout_cost: Shape (K).
        :returns perturabed_actoions: modified perturbed actions
        '''
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu
        rollout_cost = torch.zeros(K, device=self.device, dtype=self.dtype)

        state = self.state.view(1, -1).repeat(K, 1) # repeat K starting state

        stage_counter = 0
        # rollout dynamics and calculate cost for T timesteps
        k_states = []

        for t in range(T): 
            u = perturbed_actions[:,t]
            now = time.time()
            next_state = self._dynamics(state, u, t)
            # print(f"time in each rollout:{time.time() - now}")
            state = next_state # shape (K x nx)


            # if (stage_counter % self.N == 0 and stage_counter != 0): # TODO test edge cases for division

            #     start = time.time()

            #     kde = KernelDensity(bandwidth=0.3, kernel="gaussian")
            #     kde.fit(state)


            #     if state.shape[1] == 1:
            #         score = kde.score_samples(state.unsqueeze(1), batch_size=4096) # kde score samples calculate log(p(x))
            #     else:
            #         score = kde.score_samples(state, batch_size=4096)
            #     pass

            #     p_x = torch.exp(score) # calculate pdf of x

            #     inv_px = (1.0 / p_x+1e-5)**1.2 # calculate inverse of the pdf
            #     inv_px = inv_px / inv_px.sum()


            #     indices = torch.multinomial(inv_px, num_samples=self.K, replacement=True)
            #     state_new = state[indices]

            #     perturbed_actions_new = perturbed_actions[indices]
            #     perturbed_actions_new[:,t:] = self.U[t:] + self.noise_dist.sample((self.K, self.T -t))

            #     rollout_cost_new = rollout_cost[indices]

                

            #     state = state_new
            #     perturbed_actions = perturbed_actions_new
            #     rollout_cost = rollout_cost_new

            k_states.append(state)
                
                # print(f"time in resample:{time.time() - start}")

            running_cost = self._running_cost(state,u,t)
                

            rollout_cost = rollout_cost + running_cost
            stage_counter+=1
            noise = perturbed_actions - self.U

        self.k_states = torch.stack(k_states, dim=1)

        return rollout_cost, perturbed_actions, noise
    
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
        # now = time.time()
        self.shift_nominal_trajectory()

        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype,device=self.device)

        cost_total = torch.zeros(self.K, device=self.device, dtype=self.dtype)

        # Generate samples of control input sequences and get rollout cost
        perturbed_actions, noise = self.get_perturbed_actions_and_noise()
        rollout_cost, perturbed_actions, noise = self.implement_rollouts(perturbed_actions)


        # Control cost
        action_cost = self.lambda_ * noise @ self.noise_sigma_inv
        control_cost = torch.sum(self.U * action_cost, dim=(1, 2))

        # Terminal cost
        terminal_cost = 0
        if self.terminal_cost is not None:
            terminal_cost = self.terminal_cost(self.k_states[:,-1])

        # Combine all the cost
        cost_total = cost_total + control_cost + rollout_cost + terminal_cost

        # Combine to find one optimal trajectory based on total cost
        action = self.compute_optimal_control_sequence(cost_total, noise)
        self.U = self.U + action
        control_input = self.U[0]
        # print(f"time per iteration: {time.time() - now}")

        # # Unroll once for the policy for visualization
        policy_states = []
        # state = self.state.unsqueeze(0)
        # for t in range(self.T): 
        #     u = self.U[t].unsqueeze(0)
        #     next_state = self._dynamics(state, u, t)
        #     state = next_state
        #     policy_states.append(state.squeeze(0))
        # policy_states = torch.stack(policy_states, dim=0)

        return control_input, self.k_states, policy_states     

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
        action, states, policy = mppi.command(state) # get the control input from mppi based on current state
        _ = env.step(action.cpu().numpy()) # execute the control input (env return info for RL, can be discarded)
        if render:
            env.unwrapped.set_render_info(states.cpu().numpy(), policy.cpu().numpy())
            env.render()
    return