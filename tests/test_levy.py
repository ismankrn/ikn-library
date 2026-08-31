import numpy as np
import pytest

from ikn_library.algorithms.levy import levy_flight


def test_levy_has_heavy_tails():
    """Mostly small steps, rare very large ones."""
    rng = np.random.default_rng(11)
    steps = np.abs(levy_flight(rng, 20000))
    assert steps.max() > 100 * np.median(steps)


def test_lower_beta_gives_heavier_tails():
    heavy = np.abs(levy_flight(np.random.default_rng(3), 20000, beta=1.1))
    light = np.abs(levy_flight(np.random.default_rng(3), 20000, beta=1.9))
    assert heavy.max() > light.max()


def test_scale_multiplies_the_steps():
    plain = levy_flight(np.random.default_rng(5), 100)
    scaled = levy_flight(np.random.default_rng(5), 100, scale=0.01)
    np.testing.assert_allclose(scaled, 0.01 * plain, rtol=1e-12)


def test_default_scale_is_exactly_neutral():
    """scale=1.0 must not perturb results by even one ulp.

    Callers that do not scale rely on this, and multiplying by 1.0 is
    exact in IEEE 754.
    """
    a = levy_flight(np.random.default_rng(7), 500)
    b = levy_flight(np.random.default_rng(7), 500, scale=1.0)
    assert np.array_equal(a, b)


def test_scaling_is_applied_to_the_numerator():
    """Bit-identical to writing ``scale * u / |v|**(1/beta)`` inline.

    Applying the scale to the result instead would differ by a rounding
    step, which is enough to change a seeded run's whole trajectory.
    """
    scale, beta, size = 0.002, 1.5, 200
    import math
    rng = np.random.default_rng(21)
    sigma = (math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
             / (math.gamma((1.0 + beta) / 2.0) * beta
                * 2.0 ** ((beta - 1.0) / 2.0))) ** (1.0 / beta)
    u = rng.normal(0.0, sigma, size)
    v = rng.normal(0.0, 1.0, size)
    inline = scale * u / np.abs(v) ** (1.0 / beta)

    helper = levy_flight(np.random.default_rng(21), size, beta, scale=scale)
    assert np.array_equal(helper, inline)


def test_draws_from_the_generator_exactly_twice():
    """The draw order is part of the contract for seeded reproducibility.

    Checked by generator state rather than by counting calls: after the
    helper runs, its generator must sit exactly where one ``u`` draw
    followed by one ``v`` draw would leave it.
    """
    import math
    beta, size = 1.5, 10
    sigma = (math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
             / (math.gamma((1.0 + beta) / 2.0) * beta
                * 2.0 ** ((beta - 1.0) / 2.0))) ** (1.0 / beta)

    used = np.random.default_rng(1)
    levy_flight(used, size, beta)

    reference = np.random.default_rng(1)
    reference.normal(0.0, sigma, size)          # u
    reference.normal(0.0, 1.0, size)            # v

    np.testing.assert_array_equal(used.random(5), reference.random(5))


def test_same_seed_gives_the_same_steps():
    a = levy_flight(np.random.default_rng(99), 50)
    b = levy_flight(np.random.default_rng(99), 50)
    np.testing.assert_array_equal(a, b)


def test_returns_the_requested_shape():
    assert levy_flight(np.random.default_rng(0), 7).shape == (7,)


@pytest.mark.parametrize("cls_name,attr", [
    ("CuckooSearch", None),
    ("FlowerPollinationAlgorithm", None),
    ("HarrisHawksOptimization", "levy_scale"),
])
def test_algorithms_share_this_helper(cls_name, attr):
    """The three Levy users all route through one implementation."""
    from ikn_library import algorithms
    cls = getattr(algorithms, cls_name)
    algo = cls(seed=4)
    steps = algo._levy_flight(6)
    assert steps.shape == (6,)
    expected = levy_flight(np.random.default_rng(4), 6, algo.levy_exponent,
                           scale=getattr(algo, attr, 1.0) if attr else 1.0)
    np.testing.assert_array_equal(steps, expected)
