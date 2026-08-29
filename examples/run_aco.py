"""Example: minimize benchmark functions with Ant Colony Optimization."""

from ikn_library import Task
from ikn_library.problems import Ackley, Rastrigin, Sphere
from ikn_library.algorithms import AntColonyOptimization

for problem_cls in (Sphere, Rastrigin, Ackley):
    task = Task(problem=problem_cls(dimension=10), max_evals=20000)
    algo = AntColonyOptimization(population_size=30, archive_size=50, seed=42)
    best_x, best_fitness = algo.run(task)
    print(f"{problem_cls.__name__:>10}: best fitness = {best_fitness:.6g} "
          f"({task.evals} evals, {task.iters} iters)")
