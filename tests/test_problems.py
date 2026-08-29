import numpy as np
import pytest

from ikn_library.problems import Ackley, Problem, Rastrigin, Sphere


@pytest.mark.parametrize("cls", [Sphere, Rastrigin, Ackley])
def test_benchmark_optimum_is_zero(cls):
    problem = cls(dimension=10)
    assert problem.evaluate(np.zeros(10)) == pytest.approx(0.0, abs=1e-9)


def test_evaluate_rejects_wrong_shape():
    problem = Sphere(dimension=5)
    with pytest.raises(ValueError):
        problem.evaluate(np.zeros(3))


def test_custom_problem():
    class Linear(Problem):
        def __init__(self):
            super().__init__(dimension=3, lower=0.0, upper=1.0)

        def _evaluate(self, x):
            return np.sum(x)

    assert Linear().evaluate([0.1, 0.2, 0.3]) == pytest.approx(0.6)


def test_invalid_bounds_raise():
    with pytest.raises(ValueError):
        Sphere(dimension=2, lower=1.0, upper=-1.0)
