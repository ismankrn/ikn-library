"""Harris Hawks Optimization for continuous search spaces.

Reference:
    A. A. Heidari, S. Mirjalili, H. Faris, I. Aljarah, M. Mafarja, and
    H. Chen, "Harris hawks optimization: algorithm and applications,"
    Future Generation Computer Systems, 97, 849-872, 2019.
"""

import math

import numpy as np

from ikn_library.algorithms.algorithm import Algorithm


class HarrisHawksOptimization(Algorithm):
    """Harris Hawks Optimization (Heidari et al., 2019).

    Modeled on the cooperative "pounce" by which Harris's hawks hunt,
    and the most **branched** algorithm in this library: six distinct
    moves, chosen by a two-level test on how much energy the prey has
    left.

    Everything hinges on the *escaping energy*, which falls across the
    run. While ``|E| >= 1`` the prey is still strong and the hawks
    scatter to **explore**; once ``|E| < 1`` they close in, and which of
    four **besiege** moves they use depends on how much energy remains
    and on whether the prey manages to bolt:

    | Prey bolts? | Energy | Move |
    |---|---|---|
    | no | ``\\|E\\| >= 0.5`` | soft besiege |
    | no | ``\\|E\\| < 0.5`` | hard besiege |
    | yes | ``\\|E\\| >= 0.5`` | soft besiege with rapid dives |
    | yes | ``\\|E\\| < 0.5`` | hard besiege with rapid dives |

    The two *dive* moves are the interesting ones: they build two
    candidates — a direct approach and a Lévy-flight zigzag — and keep
    whichever actually improves on the hawk, or neither. They are the
    only moves here that spend extra evaluations to compare alternatives
    before committing.

    Like the Grey Wolf Optimizer, the energy schedule was tied to the
    run's progress in the original paper, so no decay had to be added.

    Args:
        population_size: Number of hawks.
        energy_start: Initial magnitude of the escaping energy, halved
            in effect by a random factor each iteration. The paper
            uses 2.
        levy_exponent: Lévy exponent ``beta`` for the dive moves.
        levy_scale: Scale of the Lévy step. The paper uses 0.01;
            0.002 measures better on shifted and rotated
            problems, see the algorithm page.
        seed: Random seed for reproducibility.
    """

    def __init__(self, population_size=30, energy_start=2.0,
                 levy_exponent=1.5, levy_scale=0.002, seed=None):
        super().__init__(population_size=population_size, seed=seed)
        if energy_start <= 0:
            raise ValueError("energy_start must be > 0")
        if not 0.0 < levy_exponent <= 2.0:
            raise ValueError("levy_exponent must be in (0, 2]")
        if levy_scale <= 0:
            raise ValueError("levy_scale must be > 0")
        self.energy_start = float(energy_start)
        self.levy_exponent = float(levy_exponent)
        self.levy_scale = float(levy_scale)

    def _progress(self, task):
        if np.isfinite(task.max_evals):
            return min(task.evals / task.max_evals, 1.0)
        if np.isfinite(task.max_iters):
            return min(task.iters / task.max_iters, 1.0)
        return 0.0

    def _levy_flight(self, size):
        """Lévy-distributed steps via Mantegna's algorithm."""
        beta = self.levy_exponent
        sigma = (math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
                 / (math.gamma((1.0 + beta) / 2.0) * beta
                    * 2.0 ** ((beta - 1.0) / 2.0))) ** (1.0 / beta)
        u = self.rng.normal(0.0, sigma, size)
        v = self.rng.normal(0.0, 1.0, size)
        return self.levy_scale * u / np.abs(v) ** (1.0 / beta)

    def _explore(self, task, hawks, i, rabbit, mean):
        """|E| >= 1: perch on a random hawk, or near the flock's centre."""
        if self.rng.random() >= 0.5:
            other = hawks[self.rng.integers(self.population_size)]
            return other - self.rng.random() * np.abs(
                other - 2.0 * self.rng.random() * hawks[i])
        span = task.upper - task.lower
        return (rabbit - mean) - self.rng.random() * (
            task.lower + self.rng.random() * span)

    def _dive(self, task, hawks, i, rabbit, mean, energy, jump, hard):
        """A direct approach and a Lévy zigzag; keep the better, if any.

        Costs two evaluations, and may keep neither candidate.
        """
        anchor = mean if hard else hawks[i]
        y = task.repair(rabbit - energy * np.abs(jump * rabbit - anchor))
        y_fitness = task.eval(y)
        if task.stopping_condition():
            return y, y_fitness

        z = task.repair(y + self.rng.random(task.dimension)
                        * self._levy_flight(task.dimension))
        z_fitness = task.eval(z)
        if z_fitness < y_fitness:
            return z, z_fitness
        return y, y_fitness

    def run_iteration(self, task, state):
        hawks, fitness = state
        rabbit = hawks[np.argmin(fitness)].copy()
        mean = hawks.mean(axis=0)
        # Escaping energy decays over the run; E0 makes it two-sided.
        envelope = self.energy_start * (1.0 - self._progress(task))

        for i in range(self.population_size):
            if task.stopping_condition():
                break

            energy = envelope * self.rng.uniform(-1.0, 1.0)

            if abs(energy) >= 1.0:
                candidate = task.repair(
                    self._explore(task, hawks, i, rabbit, mean))
                hawks[i], fitness[i] = candidate, task.eval(candidate)
                continue

            bolts = self.rng.random() < 0.5
            jump = 2.0 * (1.0 - self.rng.random())      # prey's jump strength

            if not bolts:
                if abs(energy) >= 0.5:                  # soft besiege
                    candidate = ((rabbit - hawks[i])
                                 - energy * np.abs(jump * rabbit - hawks[i]))
                else:                                    # hard besiege
                    candidate = rabbit - energy * np.abs(rabbit - hawks[i])
                candidate = task.repair(candidate)
                hawks[i], fitness[i] = candidate, task.eval(candidate)
            else:
                candidate, candidate_fitness = self._dive(
                    task, hawks, i, rabbit, mean, energy, jump,
                    hard=abs(energy) < 0.5)
                if candidate_fitness < fitness[i]:      # dives are greedy
                    hawks[i], fitness[i] = candidate, candidate_fitness

        return hawks, fitness
