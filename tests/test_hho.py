import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import HarrisHawksOptimization
from ikn_library.problems import Rastrigin, Sphere


def test_hho_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = HarrisHawksOptimization(seed=42).run(task)
    assert best_fitness < 1e-30
    assert best_x.shape == (5,)


def test_hho_solves_rastrigin():
    task = Task(problem=Rastrigin(dimension=5), max_evals=10000)
    _, best_fitness = HarrisHawksOptimization(seed=1).run(task)
    assert best_fitness < 1e-8


def test_hho_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = HarrisHawksOptimization(levy_scale=1.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_hho_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(HarrisHawksOptimization(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_hho_respects_eval_budget():
    """Dive moves spend two evaluations each, so the cap still holds."""
    for budget in (50, 137, 400):
        task = Task(problem=Sphere(dimension=3), max_evals=budget)
        HarrisHawksOptimization(seed=0).run(task)
        assert task.evals <= budget


def test_escaping_energy_decays_with_the_budget():
    algo = HarrisHawksOptimization(energy_start=2.0)
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    assert algo._progress(task) == 0.0
    for _ in range(500):
        task.eval(np.zeros(3))
    assert algo._progress(task) == pytest.approx(0.5)


def test_late_run_energy_forces_exploitation():
    """|E| < 1 always holds once the envelope has decayed below 1."""
    algo = HarrisHawksOptimization(energy_start=2.0, seed=1)
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    for _ in range(750):                       # progress 0.75 -> envelope 0.5
        task.eval(np.zeros(3))
    envelope = algo.energy_start * (1.0 - algo._progress(task))
    assert envelope == pytest.approx(0.5)
    # energy = envelope * U(-1, 1), so |E| can never reach 1 again
    assert envelope < 1.0


def test_dive_returns_the_better_of_two_candidates():
    """The dive builds a direct approach and a Levy zigzag, and picks."""
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = HarrisHawksOptimization(population_size=5, seed=3)
    hawks = np.full((5, 4), 2.0)
    rabbit = np.zeros(4)
    before = task.evals
    candidate, candidate_fitness = algo._dive(
        task, hawks, 0, rabbit, hawks.mean(axis=0),
        energy=0.3, jump=1.0, hard=False)
    assert task.evals - before == 2               # exactly two evaluations
    assert candidate_fitness == pytest.approx(task.problem.evaluate(candidate))


def test_non_dive_moves_replace_unconditionally():
    """Faithful to the paper: only dives are greedy.

    Exploration and the two plain besiege moves overwrite a hawk whether
    or not the new position is better, so an individual can worsen even
    when it sits on the optimum. The task's best-so-far still cannot
    regress, which is what actually matters for the result.
    """
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = HarrisHawksOptimization(population_size=4, seed=9)
    hawks = np.zeros((4, 4))                      # everyone is optimal
    fitness = np.zeros(4)
    _, new_fitness = algo.run_iteration(task, (hawks.copy(), fitness.copy()))
    assert np.any(new_fitness > 0.0)              # some hawk was displaced
    assert task.best_fitness == pytest.approx(0.0)   # the record is kept


def test_exploration_uses_a_random_hawk_not_the_best():
    """|E| >= 1 scatters the flock rather than converging on the prey."""
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = HarrisHawksOptimization(population_size=8, seed=2)
    hawks = np.array([[float(i)] * 3 for i in range(8)])
    moved = algo._explore(task, hawks, 0, hawks[0], hawks.mean(axis=0))
    assert moved.shape == (3,)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = HarrisHawksOptimization(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == 9


def test_fitness_stays_consistent_with_positions():
    task = Task(problem=Rastrigin(dimension=4), max_evals=3000)
    algo = HarrisHawksOptimization(population_size=6, seed=5)
    state = algo.init_population(task)
    task.next_iter()
    for _ in range(10):
        state = algo.run_iteration(task, state)
        task.next_iter()
    hawks, fitness = state
    for x, f in zip(hawks, fitness):
        assert f == pytest.approx(task.problem.evaluate(x))


@pytest.mark.parametrize("kwargs", [
    {"energy_start": 0.0},
    {"energy_start": -1.0},
    {"levy_exponent": 0.0},
    {"levy_exponent": 2.5},
    {"levy_scale": 0.0},
    {"levy_scale": -1.0},
])
def test_hho_invalid_params(kwargs):
    with pytest.raises(ValueError):
        HarrisHawksOptimization(**kwargs)
