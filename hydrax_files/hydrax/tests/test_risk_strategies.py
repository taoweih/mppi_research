import jax
import jax.numpy as jnp

from hydrax.risk import (
    AverageCost,
    BestCase,
    ConditionalValueAtRisk,
    ExponentialWeightedAverage,
    ValueAtRisk,
    WorstCase,
)


def test_risk() -> None:
    """Quick sanity check on various risk strategies."""
    rng = jax.random.key(0)

    n, m = 10, 2
    costs = jax.random.normal(rng, (n, m))

    avg = AverageCost().combine_costs(costs)
    assert avg.shape == (m,)

    worst = WorstCase().combine_costs(costs)
    assert worst.shape == (m,)

    best = BestCase().combine_costs(costs)
    assert best.shape == (m,)

    weighted = ExponentialWeightedAverage(2.0).combine_costs(costs)
    assert weighted.shape == (m,)

    var = ValueAtRisk(0.1).combine_costs(costs)
    assert var.shape == (m,)

    cvar = ConditionalValueAtRisk(0.1).combine_costs(costs)
    assert cvar.shape == (m,)

    assert jnp.all(ConditionalValueAtRisk(0.0).combine_costs(costs) == worst)
    assert jnp.all(ConditionalValueAtRisk(1.0).combine_costs(costs) == avg)

    assert jnp.all(avg <= worst)
    assert jnp.all(avg >= best)
    assert jnp.all(weighted <= worst)
    assert jnp.all(weighted >= best)
    assert jnp.all(weighted >= avg)
    assert jnp.all(var <= worst)
    assert jnp.all(var >= avg)
    assert jnp.all(cvar >= var)
    assert jnp.all(cvar <= worst)


if __name__ == "__main__":
    test_risk()
