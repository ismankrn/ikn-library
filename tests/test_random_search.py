import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import ParticleSwarmOptimization, RandomSearch
from ikn_library.problems import Problem, Sphere


class Recorder(Problem):
    """Records every point it is asked to score."""

    def __init__(self, dimension=4, constant=None):
        super().__init__(dimension=dimension, lower=-5.0, upper=5.0)
        self.seen = []
        self.constant = constant

    def _evaluate(self, x):
        self.seen.append(np.array(x, dtype=float))
        return 0.0 if self.constant is not None else float(np.sum(np.square(x)))


def test_random_search_runs_and_spends_the_budget():
    task = Task(problem=Sphere(dimension=5), max_evals=2000)
    best_x, best_fitness = RandomSearch(population_size=20, seed=42).run(task)
    assert task.evals == 2000
    assert best_x.shape == (5,)
    assert best_fitness >= 0.0


def test_random_search_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    best_x, _ = RandomSearch(seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_random_search_is_reproducible():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=500)
        results.append(RandomSearch(population_size=10, seed=3).run(task)[1])
    assert results[0] == results[1]


def test_random_search_is_blind_to_the_objective():
    """Its identity: the points it visits do not depend on what it scores.

    Any real metaheuristic steers towards good regions, so the same seed
    on two different objectives diverges after the first batch.
    """
    shaped, flat = Recorder(), Recorder(constant=0.0)
    for problem in (shaped, flat):
        RandomSearch(population_size=10, seed=5).run(
            Task(problem=problem, max_evals=200))
    np.testing.assert_allclose(np.array(shaped.seen), np.array(flat.seen))

    steered, steered_flat = Recorder(), Recorder(constant=0.0)
    for problem in (steered, steered_flat):
        ParticleSwarmOptimization(population_size=10, seed=5).run(
            Task(problem=problem, max_evals=200))
    assert not np.allclose(np.array(steered.seen), np.array(steered_flat.seen))


def test_random_search_never_exploits():
    """The last batch is no better than the first: nothing is learned."""
    problem = Recorder(dimension=6)
    task = Task(problem=problem, max_evals=4000)
    RandomSearch(population_size=40, seed=11).run(task)

    seen = np.array([float(np.sum(np.square(x))) for x in problem.seen])
    first, last = seen[:400], seen[-400:]
    # within a few percent -- sampling noise only, no downward trend
    assert abs(first.mean() - last.mean()) / first.mean() < 0.1

    # a real optimizer on the same budget does show the trend
    pso_task = Task(problem=Sphere(dimension=6), max_evals=4000)
    ParticleSwarmOptimization(population_size=40, seed=11).run(pso_task)
    assert pso_task.best_fitness < task.best_fitness / 100


def test_random_search_samples_the_whole_box():
    problem = Recorder(dimension=3)
    RandomSearch(population_size=50, seed=2).run(
        Task(problem=problem, max_evals=5000))
    points = np.array(problem.seen)
    np.testing.assert_allclose(points.mean(axis=0), 0.0, atol=0.25)
    assert np.all(points.min(axis=0) < -4.5)
    assert np.all(points.max(axis=0) > 4.5)


@pytest.mark.parametrize("population_size", [0, -1])
def test_random_search_validates_population_size(population_size):
    with pytest.raises(ValueError):
        RandomSearch(population_size=population_size)
