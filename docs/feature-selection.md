# Feature Selection

`FeatureSelectionProblem` implements wrapper-based feature selection: an
optimizer searches for the feature subset that maximizes a cross-validated
model score while keeping the subset small.

Requires scikit-learn:

```bash
pip install "ikn-library[ml]"
```

## Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier

from ikn_library import Task
from ikn_library.problems import FeatureSelectionProblem
from ikn_library.algorithms import BinaryAntColonyOptimization

X, y = load_breast_cancer(return_X_y=True)

problem = FeatureSelectionProblem(
    X, y,
    estimator=KNeighborsClassifier(n_neighbors=5),
    cv=5,
    scoring="accuracy",
    alpha=0.99,
)
task = Task(problem=problem, max_evals=1000)
algo = BinaryAntColonyOptimization(population_size=20, evaporation=0.1, seed=42)
best_x, best_fitness = algo.run(task)

print("Selected features:", problem.selected_features(best_x))
```

On the breast-cancer dataset this typically selects a handful of features
while *improving* the cross-validated accuracy over using all 30.

## The fitness function

The fitness (minimized) balances score quality against subset size:

```
alpha * (1 - cv_score) + (1 - alpha) * n_selected / n_features
```

- `alpha` close to 1 prioritizes the model score.
- Lower `alpha` presses harder for smaller subsets.
- An empty subset receives the worst possible fitness (1.0).

This weighted formulation is standard in the wrapper feature-selection
literature — see E. Emary, H. M. Zawbaa, and A. E. Hassanien, "Binary
grey wolf optimization approaches for feature selection,"
*Neurocomputing*, 172, 371–381, 2016; the same form is used in J. Too's
[Wrapper-Feature-Selection-Toolbox](https://github.com/JingweiToo/Wrapper-Feature-Selection-Toolbox)
and in NiaPy's feature-selection tutorial.

## Notes

- Solutions are vectors in `[0, 1]`; entries above `threshold` (default
  0.5) mark selected features. This means **continuous algorithms** like
  `AntColonyOptimization` can also optimize the problem, not only binary
  ones.
- Any scikit-learn estimator and scoring name works (`"f1"`,
  `"roc_auc"`, regressors with `"r2"`, ...).
- Use `problem.selected_features(best_x)` to get the selected column
  indices and `problem.feature_mask(best_x)` for a boolean mask.
