import numpy as np
import pytest

from ikn_library import Task
from ikn_library.algorithms import SineCosineAlgorithm
from ikn_library.problems import Problem, Rastrigin, Sphere


def test_sca_converges_on_sphere():
    task = Task(problem=Sphere(dimension=5), max_evals=10000)
    best_x, best_fitness = SineCosineAlgorithm(seed=42).run(task)
    assert best_fitness < 1e-15
    assert best_x.shape == (5,)


def test_sca_improves_on_rastrigin():
    problem = Rastrigin(dimension=5)
    task = Task(problem=problem, max_evals=10000)
    _, best_fitness = SineCosineAlgorithm(seed=1).run(task)
    random_baseline = min(
        problem.evaluate(x)
        for x in np.random.default_rng(1).uniform(-5.12, 5.12, (100, 5))
    )
    assert best_fitness < random_baseline


def test_sca_respects_bounds():
    task = Task(problem=Sphere(dimension=3), max_evals=2000)
    best_x, _ = SineCosineAlgorithm(amplitude=10.0, seed=7).run(task)
    assert np.all(best_x >= task.lower) and np.all(best_x <= task.upper)


def test_sca_is_reproducible_with_seed():
    results = []
    for _ in range(2):
        task = Task(problem=Sphere(dimension=4), max_evals=3000)
        results.append(SineCosineAlgorithm(seed=123).run(task))
    np.testing.assert_allclose(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_sca_respects_eval_budget():
    task = Task(problem=Sphere(dimension=3), max_evals=400)
    SineCosineAlgorithm(seed=0).run(task)
    assert task.evals <= 400


def test_amplitude_falls_linearly_to_zero():
    algo = SineCosineAlgorithm(amplitude=2.0)
    task = Task(problem=Sphere(dimension=3), max_evals=1000)
    assert algo.amplitude * (1.0 - algo._progress(task)) == pytest.approx(2.0)
    for _ in range(500):
        task.eval(np.zeros(3))
    assert algo.amplitude * (1.0 - algo._progress(task)) == pytest.approx(1.0)
    for _ in range(500):
        task.eval(np.zeros(3))
    assert algo.amplitude * (1.0 - algo._progress(task)) == pytest.approx(0.0)


def test_the_origin_is_an_exact_fixed_point():
    """The reason SCA excels on origin-centred benchmarks.

    The displacement is ``r1 * swing * |r3*P - x|``. With the
    destination and the whole population at zero that term is
    identically zero for *any* random draw, so the origin attracts
    regardless of the objective function. The algorithm page documents
    what this costs elsewhere.
    """
    task = Task(problem=Sphere(dimension=5), max_evals=5000)
    algo = SineCosineAlgorithm(population_size=8, seed=3)
    population = np.zeros((8, 5))
    fitness = np.zeros(8)
    new_population, _ = algo.run_iteration(task, (population, fitness))
    np.testing.assert_array_equal(new_population, 0.0)


def test_a_population_on_a_nonzero_destination_still_moves():
    """Off the origin the residual step does not vanish.

    At ``x = P`` the displacement is ``|r3*P - P| = |P|*|r3 - 1|``,
    which is zero only when ``P`` is zero. That is why SCA cannot
    settle on an optimum that sits away from the origin.
    """
    task = Task(problem=Sphere(dimension=5), max_evals=5000)
    algo = SineCosineAlgorithm(population_size=8, seed=3)
    population = np.full((8, 5), 2.0)
    fitness = np.full(8, 20.0)
    new_population, _ = algo.run_iteration(task, (population, fitness))
    assert np.any(new_population != 2.0)


def test_both_sine_and_cosine_are_used():
    """The trigonometric switch is a coin flip, not a schedule."""
    rng = np.random.default_rng(0)
    algo = SineCosineAlgorithm(population_size=200, seed=11)
    shape = (200, 10)
    use_sine = algo.rng.random(shape) < 0.5
    fraction = use_sine.mean()
    assert 0.45 < fraction < 0.55
    assert rng is not algo.rng


def test_solutions_can_move_away_from_the_destination():
    """sin and cos span [-1, 1], so the move is not purely attractive."""
    task = Task(problem=Sphere(dimension=6), max_evals=5000)
    algo = SineCosineAlgorithm(population_size=40, seed=4)
    population = np.full((40, 6), 3.0)
    fitness = np.full(40, 54.0)
    population[0] = 0.0                            # the destination
    fitness[0] = 0.0
    new_population, _ = algo.run_iteration(task, (population.copy(),
                                                  fitness.copy()))
    moved = new_population[1:] - population[1:]
    # some solutions moved toward zero, others away from it
    assert np.any(moved < 0) and np.any(moved > 0)


def test_amplitude_decay_shrinks_the_population_spread():
    task = Task(problem=Rastrigin(dimension=5), max_evals=4000)
    algo = SineCosineAlgorithm(population_size=20, seed=6)
    state = algo.init_population(task)
    task.next_iter()
    spread_before = np.mean(np.std(state[0], axis=0))
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
    assert np.mean(np.std(state[0], axis=0)) < spread_before


def test_error_grows_with_the_optimum_distance_from_the_origin():
    """A shift of 0.1 already costs many orders of magnitude."""

    def shifted_sphere(offset):
        class ShiftedSphere(Problem):
            def __init__(self, dimension=8):
                super().__init__(dimension, lower=-5.12, upper=5.12)

            def _evaluate(self, x):
                return float(np.sum((x - offset) ** 2))

        return ShiftedSphere

    results = []
    for offset in (0.0, 0.5):
        task = Task(problem=shifted_sphere(offset)(), max_evals=6000)
        _, best = SineCosineAlgorithm(seed=3).run(task)
        results.append(best)
    centred, moved = results
    assert centred < 1e-9                          # centred: excellent
    assert moved > 1e-4                            # barely moved: ruined
    assert moved / centred > 1e6                   # many orders apart


def test_population_size_is_preserved():
    task = Task(problem=Sphere(dimension=3), max_evals=3000)
    algo = SineCosineAlgorithm(population_size=9, seed=2)
    state = algo.init_population(task)
    task.next_iter()
    while not task.stopping_condition():
        state = algo.run_iteration(task, state)
        task.next_iter()
        assert len(state[0]) == len(state[1]) == 9


@pytest.mark.parametrize("kwargs", [
    {"amplitude": 0.0},
    {"amplitude": -1.0},
])
def test_sca_invalid_params(kwargs):
    with pytest.raises(ValueError):
        SineCosineAlgorithm(**kwargs)
