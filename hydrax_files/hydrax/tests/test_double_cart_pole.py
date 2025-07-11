import jax.numpy as jnp
from mujoco import mjx

from hydrax.tasks.double_cart_pole import DoubleCartPole


def test_double_cart_pole() -> None:
    """Make sure we can instantiate the task."""
    task = DoubleCartPole()
    assert isinstance(task, DoubleCartPole)

    state = mjx.make_data(task.model)
    state = state.replace(qpos=jnp.array([0.0, 0.1, 0.1]))  # x, θ₁, θ₂
    state = mjx.forward(task.model, state)
    tip_pos = state.site_xpos[task.tip_id]
    assert tip_pos[0] != 0.0  # x position
    assert tip_pos[1] == 0.0  # y position
    assert tip_pos[2] > 0.0  # z position

    ell = task.running_cost(state, jnp.zeros(1))
    assert ell.shape == ()

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert phi >= 0.0


if __name__ == "__main__":
    test_double_cart_pole()
