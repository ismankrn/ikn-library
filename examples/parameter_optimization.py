"""Example: tune SVM hyperparameters with continuous ACO (ACO-R).

The test set is carved out before the search starts, the scaler lives
inside the pipeline so it is refitted per fold, and the final comparison
against the default SVC is made on the test set — the search's best CV
score is the maximum of 150 candidates and is optimistically biased.

Requires scikit-learn: pip install ikn-library[ml]
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ikn_library import OptimizationType, Task
from ikn_library.algorithms import AntColonyOptimization
from ikn_library.problems import Problem


class SVMTuning(Problem):
    """Search log10(C) in [-2, 3] and log10(gamma) in [-4, 1]."""

    def __init__(self, X, y, cv):
        super().__init__(dimension=2, lower=[-2.0, -4.0], upper=[3.0, 1.0])
        self.X, self.y, self.cv = X, y, cv

    def decode(self, x):
        return {"C": 10.0 ** x[0], "gamma": 10.0 ** x[1]}

    def _evaluate(self, x):
        # Scaler inside the pipeline: refitted per fold, never sees the
        # fold it is validated on.
        model = make_pipeline(StandardScaler(), SVC(kernel="rbf", **self.decode(x)))
        return cross_val_score(model, self.X, self.y, cv=self.cv).mean()


X, y = load_breast_cancer(return_X_y=True)
X_search, X_test, y_search, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
CV = StratifiedKFold(5, shuffle=True, random_state=42)

problem = SVMTuning(X_search, y_search, cv=CV)
task = Task(problem=problem, max_evals=150,
            optimization_type=OptimizationType.MAXIMIZATION)
algo = AntColonyOptimization(population_size=10, archive_size=15, seed=42)
best_x, best_score = algo.run(task)

params = problem.decode(best_x)
tuned = make_pipeline(StandardScaler(),
                      SVC(kernel="rbf", **params)).fit(X_search, y_search)

default = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
default_cv = cross_val_score(default, X_search, y_search, cv=CV).mean()
default.fit(X_search, y_search)

print(f"Best parameters    : C={params['C']:.4f}, gamma={params['gamma']:.4f}")
print(f"CV   (search data) : default {default_cv:.4f}   tuned {best_score:.4f}")
print(f"Test (held out)    : default {default.score(X_test, y_test):.4f}   "
      f"tuned {tuned.score(X_test, y_test):.4f}")
