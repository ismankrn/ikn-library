"""Example: GEO microarray data to ACO feature selection, end to end.

Downloads GSE11223 (~26 MB, cached in ~/.ikn_library/geo/) on first run.
Requires scikit-learn: pip install ikn-library[ml]
"""

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from ikn_library import Task
from ikn_library.microarray import load_geo, top_variance
from ikn_library.problems import FeatureSelectionProblem
from ikn_library.algorithms import BinaryAntColonyOptimization

data = load_geo("GSE11223", dropna_threshold=0.1, impute="mean")
print(data)

X = top_variance(data.X, 200)
y = data.y("disease")
print("Labels:", dict(y.value_counts()))

baseline = cross_val_score(KNeighborsClassifier(), X.values, y.values, cv=3).mean()

problem = FeatureSelectionProblem(X.values, y.values, cv=3)
task = Task(problem=problem, max_evals=200)
algo = BinaryAntColonyOptimization(population_size=10, seed=42)
best_x, _ = algo.run(task)

selected = problem.selected_features(best_x)
score = cross_val_score(
    KNeighborsClassifier(), X.values[:, selected], y.values, cv=3
).mean()

print(f"All {X.shape[1]} top-variance probes: accuracy = {baseline:.4f}")
print(f"{len(selected)} ACO-selected probes  : accuracy = {score:.4f}")
