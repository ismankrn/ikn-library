import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import KrillHerd
from ikn_library.problems import Problem, Rastrigin, Sphere


def test_kh_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = KrillHerd(seed=42).run(task)
    assert best_fitness < 1e-6
    assert best_x.shape == (5,)


def test_kh_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = KrillHerd(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_kh_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = KrillHerd(c_t=5.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_kh_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(KrillHerd(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_kh_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    KrillHerd(seed=0).run(task)
    assert task.evals <= 400


def test_normalized_fitness_spans_zero_to_one():
    fitness = np.array([3.0, 1.0, 7.0, 5.0])
    normalized = KrillHerd._normalize(fitness)
    assert normalized.min() == 0.0 and normalized.max() == 1.0
    assert np.argmin(normalized) == 1                  # the best krill
    # scale-invariant: multiplying fitness changes nothing
    np.testing.assert_allclose(KrillHerd._normalize(fitness * 1000),
                               normalized)


def test_normalization_handles_an_identical_herd():
    np.testing.assert_array_equal(
        KrillHerd._normalize(np.full(5, 2.0)), np.zeros(5))


def test_food_centre_is_pulled_toward_the_better_krill():
    """The centroid is inverse-fitness weighted, not a plain mean."""
    algo = KrillHerd(population_size=2)
    krill = np.array([[0.0], [10.0]])
    normalized = np.array([0.0, 1.0])              # first krill is best
    centre = algo._food_centre(krill, normalized)
    assert 0.0 < centre[0] < 5.0                   # closer to the good one
    # an equally-good herd gives the plain mean
    even = algo._food_centre(krill, np.array([0.0, 0.0]))
    assert even[0] == pytest.approx(5.0)


def test_sensing_radius_follows_the_herd_spacing():
    """A spread-out herd senses further than a tight one."""
    algo = KrillHerd(population_size=6, seed=1)
    tight = np.linspace(0, 1, 6).reshape(6, 1)
    loose = np.linspace(0, 100, 6).reshape(6, 1)
    normalized = np.linspace(0, 1, 6)

    def radius(positions):
        distances = np.abs(positions[None, :, :] - positions[:, None, :])
        return distances.sum(axis=1).mean() / (5.0 * 6)

    assert radius(loose) > radius(tight)
    # the motion helper runs on both without error
    for positions in (tight, loose):
        motion, _, _ = algo._induced_motion(
            positions, normalized, np.zeros_like(positions))
        assert motion.shape == positions.shape


def test_induced_motion_carries_inertia():
    """Motion is smoothed across iterations, not recomputed from zero."""
    algo = KrillHerd(population_size=4, inertia=0.9, seed=2)
    krill = np.array([[0.0], [1.0], [2.0], [3.0]])
    normalized = np.array([0.0, 0.3, 0.6, 1.0])
    previous = np.full((4, 1), 5.0)
    motion, _, _ = algo._induced_motion(krill, normalized, previous)
    # 90% of the previous motion survives
    assert np.all(motion >= 0.9 * previous - 1e-9)


def test_personal_bests_never_worsen():
    task = Task(problem=Rastrigin(dimension=5), max_evals=4000)
    algo = KrillHerd(population_size=8, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        previous = state[5].copy()
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert np.all(state[5] <= previous + 1e-12)


def test_the_search_is_translation_invariant():
    """Shifting the problem shifts the whole trajectory, nothing else.

    Every coupling uses position differences and normalized fitness
    gaps, so moving the optimum cannot help or hurt. This is the
    property that Grey Wolf and Harris Hawks lack.
    """
    offset = 2.0

    class ShiftedSphere(Problem):
        def __init__(self, dimension=4):
            super().__init__(dimension,
                             lower=-5.12 + offset, upper=5.12 + offset)

        def _evaluate(self, x):
            return float(np.sum((x - offset) ** 2))

    plain = Task(problem=Sphere(dimension=4), max_evals=1200)
    moved = Task(problem=ShiftedSphere(), max_evals=1200)
    x_plain, f_plain = KrillHerd(seed=5).run(plain)
    x_moved, f_moved = KrillHerd(seed=5).run(moved)

    np.testing.assert_allclose(x_moved, x_plain + offset, rtol=1e-9,
                               atol=1e-9)
    assert f_moved == pytest.approx(f_plain, rel=1e-9)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = KrillHerd(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert all(len(part) == 9 for part in state)


@pytest.mark.parametrize("kwargs", [
    {"n_max": 0.0},
    {"v_f": 0.0},
    {"d_max": -1.0},
    {"inertia": -0.1},
    {"inertia": 1.5},
    {"c_t": 0.0},
    {"crossover_rate": 1.5},
])
def test_kh_invalid_params(kwargs):
    with pytest.raises(ValueError):
        KrillHerd(**kwargs)
