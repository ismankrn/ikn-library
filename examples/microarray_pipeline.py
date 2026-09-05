"""Example: GEO microarray data to ACO feature selection, end to end.

Downloads GSE11223 (~26 MB, cached in ~/.ikn_library/geo/) on first run.

The test set is carved out before the search, the scaler sits inside the
pipeline so it is refitted per fold, and the estimator is a linear model:
with 161 training samples and 200 probes, 5-NN scores below the
majority-class rate on this dataset.

Requires scikit-learn: pip install ikn-library[ml]
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ikn_library import Task
from ikn_library.algorithms import BinaryAntColonyOptimization
from ikn_library.microarray import load_geo, top_variance
from ikn_library.problems import FeatureSelectionProblem


def model():
    """A fresh scaler + logistic regression: the scaler is refitted per fold."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


data = load_geo("GSE11223", dropna_threshold=0.1, impute="mean")
print(data)

X = top_variance(data.X, 200)
y = data.y("disease")
print("Labels:", {k: int(v) for k, v in y.value_counts().items()})

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
)
print("train:", X_train.shape, " test:", X_test.shape)

problem = FeatureSelectionProblem(
    X_train, y_train,
    estimator=model(),
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
)
task = Task(problem=problem, max_evals=2000)
algo = BinaryAntColonyOptimization(population_size=20, seed=42)
best_x, _ = algo.run(task)

selected = problem.selected_features(best_x)
before = model().fit(X_train, y_train).score(X_test, y_test)
after = model().fit(X_train[:, selected], y_train).score(X_test[:, selected], y_test)
majority = max(np.unique(y_test, return_counts=True)[1]) / len(y_test)

print(f"majority class in test   : {majority:.4f}")
print(f"all {X.shape[1]} probes           : test accuracy = {before:.4f}")
print(f"{len(selected)} ACO-selected probes : test accuracy = {after:.4f}")
