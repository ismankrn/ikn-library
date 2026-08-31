import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import GrasshopperOptimizationAlgorithm
from ikn_library.problems import Problem, Rastrigin, Sphere


def test_goa_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = GrasshopperOptimizationAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-5
    assert best_x.shape == (5,)


def test_goa_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = GrasshopperOptimizationAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_goa_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = GrasshopperOptimizationAlgorithm(c_max=10.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_goa_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(GrasshopperOptimizationAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_goa_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    GrasshopperOptimizationAlgorithm(seed=0).run(task)
    assert task.evals <= 400


def test_the_update_is_deterministic():
    """GOA's only randomness is its initial population.

    The published update rule contains no random term at all, so from a
    given state the next state is fixed. The algorithm page discusses
    what that costs.
    """
    population = np.random.default_rng(0).uniform(-5, 5, (8, 4))
    fitness = np.array([float(np.sum(x ** 2)) for x in population])

    def step(seed):
        task = Task(problem=Sphere(dimension=4), max_evals=5000)
        algo = GrasshopperOptimizationAlgorithm(population_size=8, seed=seed)
        return algo.run_iteration(task, (population.copy(), fitness.copy()))[0]

    np.testing.assert_array_equal(step(1), step(999))


def test_the_force_law_repels_close_by_and_attracts_further_out():
    """A real comfort zone: the force must change sign."""
    algo = GrasshopperOptimizationAlgorithm(intensity=0.6,
                                            attraction_length=1.5)
    # distances are rescaled into [1, 4] before the law is applied
    near = algo._social_force(np.array([1.0]))
    far = algo._social_force(np.array([4.0]))
    assert near[0] < 0 < far[0]


def test_the_comfort_zone_sits_inside_the_rescaled_range():
    """Defaults must not degenerate into pure attraction.

    Distances are mapped into [1, 4], so a zero crossing outside that
    interval means one of the two forces never fires. The defaults are
    chosen to keep the crossing inside it.
    """
    algo = GrasshopperOptimizationAlgorithm()
    radii = np.linspace(1.0, 4.0, 2001)
    forces = algo._social_force(radii)
    assert forces.min() < 0 < forces.max()
    crossing = radii[np.argmin(np.abs(forces))]
    assert 1.0 < crossing < 4.0


def test_a_wide_attraction_length_switches_the_repulsion_off():
    """The failure mode the defaults avoid, asserted so it stays visible."""
    algo = GrasshopperOptimizationAlgorithm(intensity=0.8,
                                            attraction_length=2.5)
    forces = algo._social_force(np.linspace(1.0, 4.0, 2001))
    assert np.all(forces > 0)                  # attractive everywhere


def test_a_grasshopper_exerts_no_force_on_itself():
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = GrasshopperOptimizationAlgorithm(population_size=4, seed=2)
    # two grasshoppers at the same spot must not blow up the update
    population = np.zeros((4, 3))
    fitness = np.zeros(4)
    moved, _ = algo.run_iteration(task, (population, fitness))
    assert np.all(np.isfinite(moved))


def test_the_swarm_collapses_onto_the_target():
    task = Task(problem=Rastrigin(dimension=5), max_evals=6000)
    algo = GrasshopperOptimizationAlgorithm(population_size=12, seed=6)
    state = algo.init_population(task)
    task.next_iter()
    start = np.mean(np.linalg.norm(
        state[0] - state[0][np.argmin(state[1])], axis=1))
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
    end = np.mean(np.linalg.norm(
        state[0] - state[0][np.argmin(state[1])], axis=1))
    assert end < start / 10


def test_search_is_translation_invariant():
    """Unlike its Mirjalili siblings, GOA has no origin bias.

    The social term uses only position differences and the target is
    added as-is, so the whole update is translation-equivariant.
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
    x_plain, f_plain = GrasshopperOptimizationAlgorithm(seed=5).run(plain)
    x_moved, f_moved = GrasshopperOptimizationAlgorithm(seed=5).run(moved)
    np.testing.assert_allclose(x_moved, x_plain + offset, rtol=1e-7,
                               atol=1e-7)
    assert f_moved == pytest.approx(f_plain, rel=1e-7, abs=1e-20)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = GrasshopperOptimizationAlgorithm(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == 9


@pytest.mark.parametrize("kwargs", [
    {"population_size": 1},
    {"c_max": 0.0},
    {"c_min": 0.0},
    {"c_max": 0.5, "c_min": 1.0},
    {"intensity": 0.0},
    {"attraction_length": 0.0},
])
def test_goa_invalid_params(kwargs):
    with pytest.raises(ValueError):
        GrasshopperOptimizationAlgorithm(**kwargs)
