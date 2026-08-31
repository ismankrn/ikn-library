import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import (
    DifferentialEvolution,
    SelfAdaptiveDifferentialEvolution,
)
from ikn_library.problems import Problem, Rastrigin, Sphere


def test_jde_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = SelfAdaptiveDifferentialEvolution(seed=42).run(task)
    assert best_fitness < 1e-20
    assert best_x.shape == (5,)


def test_jde_solves_rastrigin():
    task = Task(problem=Rastrigin(dimension=5), max_evals=10000)
    _, best_fitness = SelfAdaptiveDifferentialEvolution(seed=1).run(task)
    assert best_fitness < 1e-10


def test_jde_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = SelfAdaptiveDifferentialEvolution(
        max_weight=3.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_jde_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(SelfAdaptiveDifferentialEvolution(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_jde_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    SelfAdaptiveDifferentialEvolution(seed=0).run(task)
    assert task.evals <= 400


def test_jde_reuses_the_de_operators():
    """It is DE with adaptive parameters, not a different search."""
    algo = SelfAdaptiveDifferentialEvolution
    assert issubclass(algo, DifferentialEvolution)
    assert algo._mutant is DifferentialEvolution._mutant
    assert algo._crossover is DifferentialEvolution._crossover


def test_each_individual_carries_its_own_parameters():
    task = Task(problem=Sphere(dimension=3), max_evals=5000)
    algo = SelfAdaptiveDifferentialEvolution(population_size=12, seed=3)
    _, _, weights, rates = algo.init_population(task)
    assert weights.shape == (12,) and rates.shape == (12,)
    assert np.all((weights >= algo.min_weight) & (weights <= algo.max_weight))
    assert np.all((rates >= 0.0) & (rates <= 1.0))
    assert len(np.unique(weights)) > 1


def test_proposals_stay_inside_the_weight_range():
    algo = SelfAdaptiveDifferentialEvolution(min_weight=0.2, max_weight=0.4,
                                             tau_1=1.0, tau_2=1.0, seed=5)
    for _ in range(50):
        weight, rate = algo._propose_parameters(0.3, 0.5)
        assert 0.2 <= weight <= 0.4
        assert 0.0 <= rate <= 1.0


def test_proposals_can_be_switched_off():
    algo = SelfAdaptiveDifferentialEvolution(tau_1=0.0, tau_2=0.0, seed=5)
    for _ in range(20):
        assert algo._propose_parameters(0.42, 0.37) == (0.42, 0.37)


def test_parameters_are_inherited_only_by_a_winning_trial():
    """The mechanism: selection tunes the parameters, no rule does.

    A population already sitting on the optimum can never produce a
    winning trial, so no proposal is ever kept and the parameters must
    come back unchanged.
    """
    task = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo = SelfAdaptiveDifferentialEvolution(population_size=8, tau_1=1.0,
                                             tau_2=1.0, seed=4)
    population = np.zeros((8, 4))
    fitness = np.zeros(8)
    weights = np.full(8, 0.42)
    rates = np.full(8, 0.37)
    _, _, new_weights, new_rates = algo.run_iteration(
        task, (population, fitness, weights.copy(), rates.copy()))
    # every trial ties at best, and a tie is accepted, so parameters ride
    # along; what matters is that they only ever move with their solution
    assert new_weights.shape == (8,) and new_rates.shape == (8,)

    # now a population that cannot be beaten: trials are strictly worse
    task2 = Task(problem=Sphere(dimension=4), max_evals=5000)
    algo2 = SelfAdaptiveDifferentialEvolution(population_size=8, tau_1=1.0,
                                              tau_2=1.0, seed=4)
    spread = np.eye(8, 4) * 1e-9
    fitness2 = np.full(8, -1.0)              # unbeatable: nothing improves
    _, _, w2, r2 = algo2.run_iteration(
        task2, (spread, fitness2, weights.copy(), rates.copy()))
    np.testing.assert_array_equal(w2, 0.42)
    np.testing.assert_array_equal(r2, 0.37)


def test_the_population_keeps_parameter_diversity():
    """jDE does not collapse to a single F and CR."""
    task = Task(problem=Rastrigin(dimension=5), max_evals=6000)
    algo = SelfAdaptiveDifferentialEvolution(population_size=20, seed=6)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
    _, _, weights, rates = state
    assert weights.std() > 0.01 and rates.std() > 0.01


def test_it_beats_plain_de_with_the_same_strategy():
    """Self-adaptation is worth more than the strategy it runs on."""
    problem = Rastrigin(dimension=10)
    adaptive = [SelfAdaptiveDifferentialEvolution(seed=s).run(
        Task(problem=problem, max_evals=8000))[1] for s in (1, 42)]
    fixed = [DifferentialEvolution(seed=s, strategy="rand/1").run(
        Task(problem=problem, max_evals=8000))[1] for s in (1, 42)]
    assert np.mean(adaptive) < np.mean(fixed)


def test_search_is_translation_invariant():
    offset = 2.0

    class ShiftedSphere(Problem):
        def __init__(self, dimension=4):
            super().__init__(dimension,
                             lower=-5.12 + offset, upper=5.12 + offset)

        def _evaluate(self, x):
            return float(np.sum((x - offset) ** 2))

    plain = Task(problem=Sphere(dimension=4), max_evals=1500)
    moved = Task(problem=ShiftedSphere(), max_evals=1500)
    x_plain, f_plain = SelfAdaptiveDifferentialEvolution(seed=5).run(plain)
    x_moved, f_moved = SelfAdaptiveDifferentialEvolution(seed=5).run(moved)
    np.testing.assert_allclose(x_moved, x_plain + offset, rtol=1e-8,
                               atol=1e-8)
    assert f_moved == pytest.approx(f_plain, rel=1e-8, abs=1e-20)


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = SelfAdaptiveDifferentialEvolution(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert all(len(part) == 9 for part in state)


@pytest.mark.parametrize("kwargs", [
    {"min_weight": 0.0},
    {"min_weight": -0.1},
    {"min_weight": 0.9, "max_weight": 0.5},
    {"tau_1": -0.1},
    {"tau_1": 1.5},
    {"tau_2": 1.5},
    {"strategy": "nonsense"},
])
def test_jde_invalid_params(kwargs):
    with pytest.raises(ValueError):
        SelfAdaptiveDifferentialEvolution(**kwargs)
