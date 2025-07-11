from typing import Literal, Tuple

import jax
import jax.numpy as jnp
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
        self.num_knots_per_stage = num_knots_per_stage

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPIStagedRolloutParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        return MPPIStagedRolloutParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)

    def sample_knots(self, params: MPPIStagedRolloutParams) -> Tuple[jax.Array, MPPIStagedRolloutParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                self.num_knots,
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
        
        @partial(jax.vmap, in_axes=(None, 0))
        def _rollout_fn(
           x: mjx.Data, u: jax.Array
        )-> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            '''Batched version of _scan_fn'''
            final_state, (states, costs, trace_sites) =jax.lax.scan(
                _scan_fn,  x, u
            )
            return final_state, (states, costs, trace_sites)

        final_state, (states, costs, trace_sites) = _rollout_fn(state, controls)

        final_cost = jax.vmap(self.task.terminal_cost)(final_state)
        final_trace_sites = jax.vmap(self.task.get_trace_sites)(final_state)

        costs = jnp.append(costs, final_cost[:,None], axis=1)
        trace_sites = jnp.append(trace_sites, final_trace_sites[:,None], axis=1)
        
        # jax.debug.print(f"state shape:{states.qpos.shape}\r")
        # jax.debug.print(f"controls shape:{controls.shape}\r")
        # jax.debug.print(f"knots shape:{knots.shape}\r")
        # jax.debug.print(f"costs shape:{costs.shape}\r")

        return states, Trajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
        )
