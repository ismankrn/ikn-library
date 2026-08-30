import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import CoralReefsOptimization
from ikn_library.problems import Rastrigin, Sphere


def test_cro_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = CoralReefsOptimization(seed=42).run(task)
    assert best_fitness < 1e-3
    assert best_x.shape == (5,)


def test_cro_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = CoralReefsOptimization(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_cro_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = CoralReefsOptimization(mutation_scale=1.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_cro_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(CoralReefsOptimization(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_cro_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    CoralReefsOptimization(seed=0).run(task)
    assert task.evals <= 400


def test_initial_occupation_leaves_squares_empty():
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = CoralReefsOptimization(population_size=100,
                                  initial_occupation=0.4, seed=3)
    _, fitness, occupied = algo.init_population(task)
    assert occupied.sum() == 40
    # empty squares carry no fitness and cost no evaluations
    assert np.all(np.isinf(fitness[~occupied]))
    assert task.evals == 40


def test_larva_settles_in_an_empty_square():
    task = Task(problem=Sphere(dimension=2), max_evals=5000)
    algo = CoralReefsOptimization(population_size=6, seed=1)
    reef = np.zeros((6, 2))
    fitness = np.full(6, np.inf)
    occupied = np.zeros(6, dtype=bool)
    reef, fitness, occupied = algo._settle(
        task, reef, fitness, occupied, [np.array([1.0, 1.0])])
    assert occupied.sum() == 1
    assert fitness[occupied][0] == pytest.approx(2.0)


def test_larva_cannot_displace_a_better_coral():
    """A full reef of perfect corals admits nothing."""
    task = Task(problem=Sphere(dimension=2), max_evals=5000)
    algo = CoralReefsOptimization(population_size=6, settlement_attempts=20,
                                  seed=1)
    reef = np.zeros((6, 2))
    fitness = np.zeros(6)                 # every incumbent is optimal
    occupied = np.ones(6, dtype=bool)
    new_reef, new_fitness, _ = algo._settle(
        task, reef, fitness, occupied, [np.array([3.0, 3.0])])
    np.testing.assert_array_equal(new_reef, reef)   # the larva was lost
    np.testing.assert_array_equal(new_fitness, fitness)


def test_larva_takes_over_a_worse_coral():
    task = Task(problem=Sphere(dimension=2), max_evals=5000)
    algo = CoralReefsOptimization(population_size=6, seed=1)
    reef = np.full((6, 2), 9.0)
    fitness = np.full(6, 1e9)
    occupied = np.ones(6, dtype=bool)
    _, new_fitness, occupied = algo._settle(
        task, reef, fitness, occupied, [np.array([1.0, 1.0])])
    assert occupied.all()
    assert new_fitness.min() == pytest.approx(2.0)


def test_depredation_frees_the_worst_squares():
    algo = CoralReefsOptimization(population_size=10,
                                  depredation_fraction=0.2,
                                  depredation_prob=1.0, seed=5)
    fitness = np.arange(10, dtype=float)      # index 9 is the worst
    occupied = np.ones(10, dtype=bool)
    occupied = algo._depredate(fitness, occupied)
    assert occupied.sum() == 8
    assert not occupied[9] and not occupied[8]
    assert occupied[0]                        # the best coral survives


def test_depredation_never_empties_the_reef():
    algo = CoralReefsOptimization(population_size=4,
                                  depredation_fraction=1.0,
                                  depredation_prob=1.0, seed=5)
    occupied = np.array([True, False, False, False])
    occupied = algo._depredate(np.array([1.0, np.inf, np.inf, np.inf]),
                               occupied)
    assert occupied.sum() == 1


def test_budding_is_off_by_default():
    """Cloning the best corals collapses crossover; see the docs."""
    assert CoralReefsOptimization().asexual_fraction == 0.0


def test_reef_capacity_is_never_exceeded():
    task = Task(problem=Sphere(dimension=3), max_evals=6000)
    algo = CoralReefsOptimization(population_size=15, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == len(state[2]) == 15
        assert 1 <= state[2].sum() <= 15


@pytest.mark.parametrize("kwargs", [
    {"initial_occupation": 0.0},
    {"initial_occupation": 1.5},
    {"broadcast_fraction": 1.5},
    {"asexual_fraction": -0.1},
    {"depredation_fraction": 1.5},
    {"depredation_prob": 1.5},
    {"settlement_attempts": 0},
    {"mutation_scale": 0.0},
])
def test_cro_invalid_params(kwargs):
    with pytest.raises(ValueError):
        CoralReefsOptimization(**kwargs)
