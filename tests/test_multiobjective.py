import numpy as np
import pytest

from ikn_library.algorithms import NSGA2
from ikn_library.multiobjective import (
    MultiObjectiveProblem,
    MultiObjectiveTask,
    crowding_distance,
    dominates,
    non_dominated_sort,
    pareto_front,
)

# --- Pareto utilities -------------------------------------------------

def test_dominates():
    assert dominates([1, 1], [2, 2])            # better in both
    assert dominates([1, 2], [1, 3])            # equal in one, better in one
    assert not dominates([1, 3], [2, 2])        # a trade-off: neither wins
    assert not dominates([1, 1], [1, 1])        # identical: no domination


def test_non_dominated_sort_layers():
    objectives = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0],
                           [3.0, 4.0], [5.0, 5.0]])
    fronts = non_dominated_sort(objectives)
    assert sorted(fronts[0].tolist()) == [0, 1, 2, 3]   # the trade-off curve
    assert fronts[1].tolist() == [4]
    assert fronts[2].tolist() == [5]
    assert sum(len(f) for f in fronts) == len(objectives)


def test_non_dominated_sort_edge_cases():
    assert non_dominated_sort(np.empty((0, 2))) == []
    single = non_dominated_sort(np.array([[1.0, 2.0]]))
    assert len(single) == 1 and single[0].tolist() == [0]
    # identical points never dominate each other -> one front
    same = non_dominated_sort(np.ones((4, 2)))
    assert len(same) == 1 and len(same[0]) == 4


def test_crowding_distance_favours_isolated_points():
    objectives = np.array([[0.0, 1.0], [0.1, 0.9], [0.5, 0.5], [1.0, 0.0]])
    distance = crowding_distance(objectives)
    assert np.isinf(distance[0]) and np.isinf(distance[-1])   # boundaries
    assert distance[2] > distance[1]        # the lonelier interior point wins


def test_pareto_front_deduplicates():
    solutions = np.arange(8).reshape(4, 2).astype(float)
    objectives = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    front_solutions, front_objectives = pareto_front(solutions, objectives)
    assert len(front_objectives) == 2       # duplicate collapsed, [3,3] dominated
    assert len(front_solutions) == 2
    # sorted by the first objective
    assert front_objectives[0][0] < front_objectives[1][0]
    kept_all = pareto_front(solutions, objectives, unique=False)[1]
    assert len(kept_all) == 3


# --- Problem and task -------------------------------------------------

class TwoObjective(MultiObjectiveProblem):
    """f1 = sum(x), f2 = sum((x - 1)^2): a genuine trade-off."""

    def __init__(self, dimension=3):
        super().__init__(dimension, n_objectives=2, lower=0.0, upper=1.0,
                         objective_names=["sum", "distance"])

    def _evaluate(self, x):
        return np.array([np.sum(x), np.sum((x - 1.0) ** 2)])


def test_problem_returns_objective_vector():
    problem = TwoObjective(3)
    values = problem.evaluate(np.zeros(3))
    np.testing.assert_allclose(values, [0.0, 3.0])
    assert problem.objective_names == ["sum", "distance"]


def test_problem_validation():
    with pytest.raises(ValueError):
        MultiObjectiveProblem(3, n_objectives=1)
    with pytest.raises(ValueError):
        MultiObjectiveProblem(3, n_objectives=2, objective_names=["only_one"])

    class Broken(MultiObjectiveProblem):
        def __init__(self):
            super().__init__(2, n_objectives=2)

        def _evaluate(self, x):
            return np.array([1.0])          # wrong length

    with pytest.raises(ValueError):
        Broken().evaluate(np.zeros(2))


def test_task_tracks_pareto_archive():
    task = MultiObjectiveTask(problem=TwoObjective(3), max_evals=10)
    task.eval(np.zeros(3))                  # [0, 3]
    task.eval(np.ones(3))                   # [3, 0]
    task.eval(np.full(3, 0.5))              # [1.5, 0.75]
    task.eval(np.ones(3) * 0.9)             # dominated by none? check count
    solutions, objectives = task.result()
    assert task.evals == 4
    assert len(solutions) == len(objectives)
    # every archived point must be mutually non-dominated
    for a in objectives:
        for b in objectives:
            assert not dominates(a, b) or np.allclose(a, b)


def test_task_requires_a_budget_and_valid_archive():
    with pytest.raises(ValueError):
        MultiObjectiveTask(problem=TwoObjective())
    with pytest.raises(ValueError):
        MultiObjectiveTask(problem=TwoObjective(), max_evals=10, archive_size=1)


def test_task_respects_budget_and_repairs():
    task = MultiObjectiveTask(problem=TwoObjective(3), max_evals=3)
    for _ in range(6):
        task.eval(np.zeros(3))
    assert task.evals == 3 and task.stopping_condition()
    np.testing.assert_allclose(task.repair(np.array([-5.0, 0.5, 5.0])),
                               [0.0, 0.5, 1.0])


def test_archive_stays_bounded():
    task = MultiObjectiveTask(problem=TwoObjective(3), max_evals=2000,
                              archive_size=20)
    rng = np.random.default_rng(0)
    while not task.stopping_condition():
        task.eval(rng.random(3))
    assert len(task.result()[0]) <= 40      # pruned well below the raw count


# --- NSGA-II ----------------------------------------------------------

def test_nsga2_finds_a_spread_front():
    task = MultiObjectiveTask(problem=TwoObjective(5), max_evals=3000)
    _, objectives = NSGA2(population_size=30, seed=42).run(task)
    assert len(objectives) > 5
    # the front must span a real range of the first objective
    assert objectives[:, 0].max() - objectives[:, 0].min() > 1.0
    # and be internally non-dominated
    assert len(non_dominated_sort(objectives)[0]) == len(objectives)


def test_nsga2_approximates_a_known_front():
    """ZDT1's true front is f2 = 1 - sqrt(f1)."""

    class ZDT1(MultiObjectiveProblem):
        def __init__(self):
            super().__init__(6, n_objectives=2, lower=0.0, upper=1.0)

        def _evaluate(self, x):
            g = 1.0 + 9.0 * np.mean(x[1:])
            return np.array([x[0], g * (1.0 - np.sqrt(x[0] / g))])

    task = MultiObjectiveTask(problem=ZDT1(), max_evals=8000)
    _, objectives = NSGA2(population_size=40, seed=1).run(task)
    error = np.abs(objectives[:, 1] - (1.0 - np.sqrt(objectives[:, 0])))
    assert error.mean() < 0.05
    assert objectives[:, 0].min() < 0.1 and objectives[:, 0].max() > 0.9


def test_nsga2_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = MultiObjectiveTask(problem=TwoObjective(4), max_evals=1000)
        results.append(NSGA2(population_size=20, seed=7).run(task)[1])
    np.testing.assert_allclose(results[0], results[1])


def test_nsga2_respects_eval_budget():
    task = MultiObjectiveTask(problem=TwoObjective(3), max_evals=300)
    NSGA2(population_size=20, seed=0).run(task)
    assert task.evals <= 300


@pytest.mark.parametrize("kwargs", [
    {"crossover_rate": 1.5},
    {"mutation_rate": -0.1},
    {"mutation_scale": 0.0},
    {"blend_alpha": -1.0},
])
def test_nsga2_invalid_params(kwargs):
    with pytest.raises(ValueError):
        NSGA2(**kwargs)
