import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import ParticleSwarmOptimization
from ikn_library.problems import Problem, Rastrigin, Sphere


def test_pso_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = ParticleSwarmOptimization(seed=42).run(task)
    assert best_fitness < 1e-20
    assert best_x.shape == (5,)


def test_pso_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = ParticleSwarmOptimization(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_pso_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = ParticleSwarmOptimization(max_velocity=5.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_pso_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(ParticleSwarmOptimization(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_pso_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    ParticleSwarmOptimization(seed=0).run(task)
    assert task.evals <= 400


def test_particles_start_at_rest():
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = ParticleSwarmOptimization(population_size=6, seed=3)
    _, _, velocities, _, _ = algo.init_population(task)
    np.testing.assert_array_equal(velocities, 0.0)


def test_inertia_falls_linearly_with_the_budget():
    algo = ParticleSwarmOptimization(w_start=0.7, w_end=0.4)
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    assert algo._inertia(task) == pytest.approx(0.7)
    for _ in range(500):
        task.eval(np.zeros(3))
    assert algo._inertia(task) == pytest.approx(0.55)
    for _ in range(500):
        task.eval(np.zeros(3))
    assert algo._inertia(task) == pytest.approx(0.4)


def test_velocity_is_clamped():
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = ParticleSwarmOptimization(population_size=6, max_velocity=0.05,
                                     c1=50.0, c2=50.0, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    limit = 0.05 * (task.upper - task.lower)
    for _ in range(5):
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert np.all(np.abs(state[2]) <= limit + 1e-12)


def test_a_settled_swarm_does_not_drift():
    """Zero velocity with every particle on both attractors is a fixed point."""
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = ParticleSwarmOptimization(population_size=5, seed=1)
    particles = np.zeros((5, 4))
    fitness = np.zeros(5)
    velocities = np.zeros((5, 4))
    new_particles, _, new_velocities, _, _ = algo.run_iteration(
        task, (particles, fitness, velocities, particles.copy(),
               fitness.copy()))
    np.testing.assert_allclose(new_velocities, 0.0, atol=1e-12)
    np.testing.assert_allclose(new_particles, 0.0, atol=1e-12)


def test_personal_bests_never_worsen():
    task = Task(problem=Rastrigin(dimension=5), max_evals=4000)
    algo = ParticleSwarmOptimization(population_size=8, seed=4)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        previous = state[4].copy()
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert np.all(state[4] <= previous + 1e-12)


def test_cognitive_only_swarm_ignores_the_global_best():
    """With c2=0 each particle searches around its own history alone."""
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = ParticleSwarmOptimization(population_size=6, c1=2.0, c2=0.0,
                                     seed=5)
    state = algo.init_population(task)
    task.next_iter()
    spread_before = np.mean(np.std(state[0], axis=0))
    for _ in range(20):
        state = algo.run_iteration(task, state)
        task.next_iter()
    # without a social pull the swarm does not collapse to a point
    assert np.mean(np.std(state[0], axis=0)) > spread_before / 100


def test_social_pull_contracts_the_swarm():
    task = Task(problem=Sphere(dimension=3), max_evals=4000)
    algo = ParticleSwarmOptimization(population_size=10, seed=6)
    state = algo.init_population(task)
    task.next_iter()
    spread_before = np.mean(np.std(state[0], axis=0))
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
    assert np.mean(np.std(state[0], axis=0)) < spread_before


def test_search_is_translation_invariant():
    """Both pulls are differences, so moving the optimum is free.

    PSO is translation-invariant but *not* rotation-invariant; see the
    algorithm page.
    """
    offset = 2.0

    class ShiftedSphere(Problem):
        def __init__(self, dimension=4):
            super().__init__(dimension,
                             lower=-5.12 + offset, upper=5.12 + offset)

        def _evaluate(self, x):
            return float(np.sum((x - offset) ** 2))

    plain = Task(problem=Sphere(dimension=4), max_evals=1500)
    moved = Task(problem=ShiftedSphere(), max_evals=1500)
    x_plain, f_plain = ParticleSwarmOptimization(seed=5).run(plain)
    x_moved, f_moved = ParticleSwarmOptimization(seed=5).run(moved)

    np.testing.assert_allclose(x_moved, x_plain + offset, rtol=1e-8,
                               atol=1e-8)
    assert f_moved == pytest.approx(f_plain, rel=1e-8, abs=1e-20)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = ParticleSwarmOptimization(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert all(len(part) == 9 for part in state)


@pytest.mark.parametrize("kwargs", [
    {"w_start": -0.1},
    {"w_end": -0.1},
    {"w_start": 0.4, "w_end": 0.9},
    {"c1": -1.0},
    {"c2": -1.0},
    {"c1": 0.0, "c2": 0.0},
    {"max_velocity": 0.0},
])
def test_pso_invalid_params(kwargs):
    with pytest.raises(ValueError):
        ParticleSwarmOptimization(**kwargs)
