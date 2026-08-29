import numpy as np
import pytest

from ikn_library import OptimizationType, Task
from ikn_library.problems import Sphere


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
