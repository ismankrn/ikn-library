"""Base class for metaheuristic algorithms."""

import numpy as np


class Algorithm:
    """Base class for population-based metaheuristic algorithms.

    Subclasses implement :meth:`init_population` and :meth:`run_iteration`;
    the shared :meth:`run` loop handles the budget and result reporting.

    Args:
        population_size: Number of individuals in the population.
        seed: Random seed for reproducibility (optional).
    """

    def __init__(self, population_size=25, seed=None):
        if population_size < 1:
            raise ValueError("population_size must be >= 1")
        self.population_size = int(population_size)
        self.rng = np.random.default_rng(seed)

    def run(self, task):
        """Run the algorithm on ``task`` until its budget is exhausted.

        Returns:
            tuple: ``(best_x, best_fitness)``.
        """
        state = self.init_population(task)
        task.next_iter()
        while not task.stopping_condition():
            state = self.run_iteration(task, state)
            task.next_iter()
        return task.result()

    def init_population(self, task):
        """Create the initial population; return the algorithm state.

        The default creates ``population_size`` uniform random solutions
        and returns ``(population, fitness)`` arrays.
        """
        population = self.rng.uniform(
            task.lower, task.upper, (self.population_size, task.dimension)
        )
        fitness = np.array([task.eval(x) for x in population])
        return population, fitness

    def run_iteration(self, task, state):
        """Perform one iteration; receive and return the algorithm state."""
        raise NotImplementedError

    @property
    def name(self):
        return type(self).__name__
