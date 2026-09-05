"""Example: metaheuristic undersampling of an imbalanced dataset.

The majority class is deliberately polluted with mislabeled samples —
the regime where *which* samples you discard matters most. Protocol:
the selection is optimized against the validation split (never
undersampled), and all final numbers are reported on an untouched test
split.

Requires scikit-learn: pip install ikn-library[ml]
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library import Task
from ikn_library.algorithms import BinaryAntColonyOptimization
from ikn_library.sampling import UndersamplingProblem

# Imbalanced dataset with label noise in the majority class: 25% of the
# minority samples are mislabeled as majority.
X, y = make_classification(n_samples=3000, n_features=10, n_informative=4,
                           weights=[0.85, 0.15], flip_y=0.0, random_state=0)
rng = np.random.default_rng(0)
minority_idx = np.flatnonzero(y == 1)
mislabeled = rng.choice(minority_idx, int(0.25 * len(minority_idx)), replace=False)
y = y.copy()
y[mislabeled] = 0

X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=42)
print(f"Train class counts: minority={np.sum(y_train == 1)}, "
      f"majority={np.sum(y_train == 0)}")


def knn():
    """A fresh scaler + KNN pipeline: the scaler is refitted per subset."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


def test_f1(X_fit, y_fit):
    return f1_score(y_test, knn().fit(X_fit, y_fit).predict(X_test))


problem = UndersamplingProblem(
    X_train, y_train, X_val, y_val,
    estimator=knn(),
    target_ratio=1.0, metric="f1",
)
task = Task(problem=problem, max_evals=3000)
algo = BinaryAntColonyOptimization(population_size=20, seed=42)
best_x, best_fitness = algo.run(task)

full = test_f1(X_train, y_train)
random_scores = [
    test_f1(*problem.resampled_data(np.random.default_rng(7 + i).random(problem.dimension)))
    for i in range(5)
]
optimized = test_f1(*problem.resampled_data(best_x))

print(f"No undersampling (imbalanced) : test F1 = {full:.4f}")
print(f"Random undersampling (mean/5) : test F1 = {np.mean(random_scores):.4f}")
print(f"Optimized undersampling       : test F1 = {optimized:.4f}")
print(f"Reduced training set: {len(problem.selected_indices(best_x))} samples "
      f"(from {len(X_train)})")
