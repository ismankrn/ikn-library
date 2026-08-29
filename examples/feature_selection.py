"""Example: feature selection on the breast-cancer dataset with Binary ACO.

Requires scikit-learn: pip install ikn-library[ml]
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from ikn_library import Task
from ikn_library.problems import FeatureSelectionProblem
from ikn_library.algorithms import BinaryAntColonyOptimization

data = load_breast_cancer()
X, y = data.data, data.target
estimator = KNeighborsClassifier(n_neighbors=5)

problem = FeatureSelectionProblem(X, y, estimator=estimator, cv=5, alpha=0.99)
task = Task(problem=problem, max_evals=1000)
algo = BinaryAntColonyOptimization(population_size=20, evaporation=0.1, seed=42)
best_x, best_fitness = algo.run(task)

selected = problem.selected_features(best_x)
baseline = cross_val_score(estimator, X, y, cv=5).mean()
score = cross_val_score(estimator, X[:, selected], y, cv=5).mean()

print(f"All {X.shape[1]} features : accuracy = {baseline:.4f}")
print(f"Selected {len(selected)} features: accuracy = {score:.4f}")
print("Selected feature names:", list(data.feature_names[selected]))
