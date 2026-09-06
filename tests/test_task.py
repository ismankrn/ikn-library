import numpy as np
import pytest

from ikn_library import OptimizationType, Task
from ikn_library.algorithms import AntColonyOptimization
from ikn_library.problems import Problem, Sphere


def test_requires_a_budget():
    with pytest.raises(ValueError):
        Task(problem=Sphere(dimension=2))


def test_counts_evals_and_tracks_best():
    task = Task(problem=Sphere(dimension=2), max_evals=10)
    task.eval(np.array([1.0, 1.0]))
    task.eval(np.array([0.5, 0.5]))
    assert task.evals == 2
    assert task.best_fitness == pytest.approx(0.5)
    np.testing.assert_allclose(task.best_x, [0.5, 0.5])


def test_stops_at_max_evals():
    task = Task(problem=Sphere(dimension=2), max_evals=3)
    for _ in range(5):
        task.eval(np.array([1.0, 1.0]))
    assert task.evals == 3
    assert task.stopping_condition()


def test_repair_clips_to_bounds():
    task = Task(problem=Sphere(dimension=2), max_evals=10)
    repaired = task.repair(np.array([100.0, -100.0]))
    np.testing.assert_allclose(repaired, [5.12, -5.12])


def test_maximization_result_sign():
    task = Task(
        problem=Sphere(dimension=2),
        max_evals=10,
        optimization_type=OptimizationType.MAXIMIZATION,
    )
    task.eval(np.array([2.0, 0.0]))
    _, best = task.result()
    assert best == pytest.approx(4.0)


# --- patience / early stopping ----------------------------------------

class _Flat(Problem):
    """Every solution scores the same, so nothing ever improves."""

    def __init__(self):
        super().__init__(dimension=2, lower=0.0, upper=1.0)

    def _evaluate(self, x):
        return 1.0


def test_patience_defaults_to_disabled():
    task = Task(problem=Sphere(dimension=2), max_iters=5)
    assert task.patience is None
    for _ in range(5):
        task.next_iter()
    assert task.stalled_iters == 5      # counted, but never acted on
    assert not task.stopped_early


def test_patience_stops_a_stalled_run():
    task = Task(problem=_Flat(), max_iters=100, patience=3)
    AntColonyOptimization(population_size=5, seed=0).run(task)
    # one iteration finds the (constant) best, three more make no progress
    assert task.iters == 4
    assert task.stalled_iters == 3
    assert task.stopped_early
    assert task.iters < task.max_iters


def test_patience_resets_on_improvement():
    task = Task(problem=Sphere(dimension=2), max_iters=50, patience=2)
    for value, expected in ((10.0, 0), (10.0, 1), (9.0, 0), (9.0, 1), (9.0, 2)):
        task.best_fitness = value
        task.next_iter()
        assert task.stalled_iters == expected
    assert task.stopping_condition()


def test_min_delta_ignores_improvements_that_are_too_small():
    task = Task(problem=Sphere(dimension=2), max_iters=50,
                patience=2, min_delta=1.0)
    for value in (10.0, 9.6, 9.2):          # steps of 0.4 do not count
        task.best_fitness = value
        task.next_iter()
    assert task.stalled_iters == 2
    assert task.stopping_condition()

    task.best_fitness = 8.8                 # 1.2 below the last improvement
    task.next_iter()
    assert task.stalled_iters == 0


def test_budget_still_wins_and_is_not_reported_as_early():
    task = Task(problem=_Flat(), max_iters=2, patience=10)
    AntColonyOptimization(population_size=5, seed=0).run(task)
    assert task.iters == 2
    assert task.budget_exhausted
    assert not task.stopped_early


def test_patience_that_never_triggers_leaves_the_run_untouched():
    plain = Task(problem=Sphere(dimension=5), max_evals=2000)
    patient = Task(problem=Sphere(dimension=5), max_evals=2000, patience=500)
    _, plain_best = AntColonyOptimization(population_size=10, seed=1).run(plain)
    _, patient_best = AntColonyOptimization(population_size=10, seed=1).run(patient)
    assert plain.evals == patient.evals
    assert plain_best == patient_best
    assert not patient.stopped_early


@pytest.mark.parametrize("kwargs", [
    {"patience": 0},
    {"patience": -1},
    {"min_delta": -0.1},
])
def test_patience_validation(kwargs):
    with pytest.raises(ValueError):
        Task(problem=Sphere(dimension=2), max_iters=10, **kwargs)
