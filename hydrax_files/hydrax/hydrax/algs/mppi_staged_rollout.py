from typing import Literal, Tuple

import jax
import jax.numpy as jnp
import math
from jax.scipy.stats import gaussian_kde
import numpy as np

from flax.struct import dataclass

from functools import partial
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class MPPIStagedRolloutParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """


class MPPIStagedRollout(SamplingBasedController):
    """Model-predictive path integral control.

    Implements "MPPI-generic" as described in https://arxiv.org/abs/2409.07563.
    Unlike the original MPPI derivation, this does not assume stochastic,
    control-affine dynamics or a separable cost function that is quadratic in
    control.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        num_knots_per_stage: int = 4,
        kde_bandwidth: float = 1.0,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature

        self.params = None

        self.num_knots_per_stage = num_knots_per_stage
        self.kde_bandwidth = kde_bandwidth

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPIStagedRolloutParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        self.params = MPPIStagedRolloutParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)
        return MPPIStagedRolloutParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)

    def sample_knots(self, params: MPPIStagedRolloutParams) -> Tuple[jax.Array, MPPIStagedRolloutParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                #self.num_knots,
                params.mean.shape[0],
                self.task.model.nu,
            ),
        )
        controls = params.mean + self.noise_level * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPIStagedRolloutParams, rollouts: Trajectory
    ) -> MPPIStagedRolloutParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)
    
    def eval_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
    ) -> Tuple[mjx.Data, Trajectory]:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀.
            controls: The control sequences, (num rollouts, H, nu).
            knots: The control spline knots, (num rollouts, num_knots, nu).

        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, costs, and trace sites.
        """

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            x = x.replace(ctrl=u)
            x = mjx.step(model, x)  # step model + compute site positions
            cost = self.dt * self.task.running_cost(x, u)
            sites = self.task.get_trace_sites(x)
            return x, (x, cost, sites)
        
        @partial(jax.vmap, in_axes=(0, 0))
        def _rollout_fn(
           x: mjx.Data, u: jax.Array
        )-> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            '''Batched version of _scan_fn'''
            final_state, (states, costs, trace_sites) =jax.lax.scan(
                _scan_fn,  x, u
            )
            return final_state, (states, costs, trace_sites)
        
        #### TODO rollout and resample start ####

        ## Initilize full states and costs that will be updated after each stage
        states = jax.tree_util.tree_map(lambda x: jnp.zeros((self.num_samples, self.ctrl_steps)+x.shape, dtype=x.dtype),state)
        costs = jnp.zeros((self.num_samples, self.ctrl_steps))
        trace_sites = jnp.zeros((self.num_samples, self.ctrl_steps) + state.site_xpos.shape)

        # Calculate some parameters for ease of use
        num_stages = int(math.floor(self.num_knots / self.num_knots_per_stage))
        timesteps_per_stage = int(math.floor(self.ctrl_steps / self.num_knots))*self.num_knots_per_stage

        # batch init state 
        curr_state = jax.tree_util.tree_map((lambda x: jnp.repeat(x[None, ...], self.num_samples, axis=0)), state)


        for n in range(num_stages-1):
            # partial rollout
            partial_controls = controls[:,n*timesteps_per_stage:(n+1)*timesteps_per_stage,:]
            latest_state, (partial_states, partial_costs, partial_trace_sites) = _rollout_fn(curr_state, partial_controls)
            costs = costs.at[:,n*timesteps_per_stage:(n+1)*timesteps_per_stage].set(partial_costs)
            trace_sites = trace_sites.at[:,n*timesteps_per_stage:(n+1)*timesteps_per_stage].set(partial_trace_sites)
            states = jax.tree_util.tree_map(lambda x, new: x.at[:, n*timesteps_per_stage:(n+1)*timesteps_per_stage,...].set(new),states, partial_states)

            # resampling indices
            jnp_latest_state = jnp.concatenate([latest_state.qpos, latest_state.qvel],axis=1)
            kde = gaussian_kde(jnp_latest_state,bw_method=self.kde_bandwidth)

            p_x = kde.pdf(jnp_latest_state)
            inv_px = (1.0 / p_x+1e-5)**1.2
            inv_px = inv_px / inv_px.sum()
            
            indices = jax.random.categorical(jax.random.PRNGKey(0),jnp.log(inv_px),shape=(self.num_samples,))

            # reorder things around (only need to reorder up to current steps but won't matter since the later ones will be overwritten)
            # TODO reorder states (in mjxData form), controls, knots, costs
            states = jax.tree_util.tree_map(lambda x: x[indices,...], states)
            controls = controls[indices,...]
            knots = knots[indices,...]
            costs = costs[indices,...]
            trace_sites = trace_sites[indices,...]

            curr_state = jax.tree_util.tree_map(lambda x: x[:,-1,...], states)

            # sample new knots, update controls
            partial_param = self.params.replace(mean= self.params.mean[(n+1)*self.num_knots_per_stage:,:])
            sampled_partial_knots, _ = self.sample_knots(partial_param)
            sampled_partial_knots = jnp.clip(
                sampled_partial_knots, self.task.u_min, self.task.u_max
            )
            knots = knots.at[:,(n+1)*self.num_knots_per_stage:,:].set(sampled_partial_knots)
            tk = partial_param.tk
            tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
            controls = self.interp_func(tq, tk, knots)

        # rollout remaining control
        partial_controls = controls[:,(num_stages-1)*timesteps_per_stage:,:]
        final_state, (partial_states, partial_costs, partial_trace_sites) = _rollout_fn(curr_state, partial_controls)
        costs = costs.at[:,(num_stages-1)*timesteps_per_stage:].set(partial_costs)
        trace_sites = trace_sites.at[:,(num_stages-1)*timesteps_per_stage:].set(partial_trace_sites)
        states = jax.tree_util.tree_map(lambda x, new: x.at[:,(num_stages-1)*timesteps_per_stage:,...].set(new),states, partial_states)

        #### rollout and resample end ####

        final_cost = jax.vmap(self.task.terminal_cost)(final_state)
        final_trace_sites = jax.vmap(self.task.get_trace_sites)(final_state)

        costs = jnp.append(costs, final_cost[:,None], axis=1)
        trace_sites = jnp.append(trace_sites, final_trace_sites[:,None], axis=1)
        
        # jax.debug.print(f"states shape:{states.qpos.shape}\r")
        # jax.debug.print(f"controls shape:{controls.shape}\r")
        # jax.debug.print(f"knots shape:{knots.shape}\r")
        # jax.debug.print(f"costs shape:{costs.shape}\r")

        return states, Trajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
        )
