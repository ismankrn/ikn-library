"""Example: tune SVM hyperparameters with continuous ACO (ACO-R).

Requires scikit-learn: pip install ikn-library[ml]
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ikn_library import OptimizationType, Task
from ikn_library.problems import Problem
from ikn_library.algorithms import AntColonyOptimization


class SVMTuning(Problem):
    """Search log10(C) in [-2, 3] and log10(gamma) in [-4, 1]."""

    def __init__(self, X, y, cv=5):
        super().__init__(dimension=2, lower=[-2.0, -4.0], upper=[3.0, 1.0])
        self.X, self.y, self.cv = X, y, cv

    def decode(self, x):
        return {"C": 10.0 ** x[0], "gamma": 10.0 ** x[1]}

    def _evaluate(self, x):
        model = SVC(kernel="rbf", **self.decode(x))
        return cross_val_score(model, self.X, self.y, cv=self.cv).mean()


X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)

problem = SVMTuning(X, y, cv=5)
task = Task(problem=problem, max_evals=150,
            optimization_type=OptimizationType.MAXIMIZATION)
algo = AntColonyOptimization(population_size=10, archive_size=15, seed=42)
best_x, best_score = algo.run(task)

baseline = cross_val_score(SVC(kernel="rbf"), X, y, cv=5).mean()
print(f"Default SVC        : accuracy = {baseline:.4f}")
print(f"Tuned SVC          : accuracy = {best_score:.4f}")
print(f"Best parameters    : {problem.decode(best_x)}")
