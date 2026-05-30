# milgboost

**Multiple Instance Learning** for **Gradient Boosting Models**.

MIL is a weakly supervised learning paradigm where labels are available for _bags_ (groups of instances) rather than individual instances. `milgboost` brings MIL to gradient boosting by wrapping LightGBM and XGBoost with custom differentiable objectives — currently the **LogSumExp Binary Cross-Entropy (LSE-BCE)** loss, a smooth approximation of the max-instance MIL loss.

## Installation

```shell
uv add milgboost
```

### Extra options

Install with a specific boosting backend:

```shell
uv add milgboost[xgboost-cpu]
uv add milgboost[xgboost] # GPU enabled
uv add milgboost[lightgbm]
uv add milgboost[xgboost-cpu,lightgbm]
```

## Module overview

| Module                     | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
| `milgboost.types`          | `Bag` / `LabeledBag` dataclasses + array↔bag conversion helpers |
| `milgboost.datasets`       | `make_mil_data()` — synthetic MIL data generator                |
| `milgboost.model.base`     | `BaseMILModel` abstract class (fit / predict / predict_proba)   |
| `milgboost.model.xgboost`  | `XGBoostMILModel` — XGBoost-backed MIL classifier               |
| `milgboost.model.lightgbm` | `LightGBMMILModel` — LightGBM-backed MIL classifier             |
| `milgboost.objective.base` | `BaseMILObjective` abstract interface for custom MIL objectives |
| `milgboost.objective.lse`  | `LSEBCE` — LogSumExp binary cross-entropy objective             |

## Output ordering

All prediction methods (`predict`, `predict_proba`, `predict_bags`, `predict_proba_bags`) return results **sorted by bag_id in ascending order**. For example, if your bag IDs are `[3, 1, 2]`, the output will be ordered as bags `[1, 2, 3]`.

**Recommendation**: Sort both `x` and `z` by `z` values before prediction to ensure output aligns with your expected ordering:

```python
# Sort x and z by z values before prediction
sort_idx = np.argsort(z)
x_sorted, z_sorted = x[sort_idx], z[sort_idx]

# Predictions will follow the sorted order
probs = model.predict_proba(x_sorted, z_sorted)
# probs[i] corresponds to bag i (after sorting)
```

Using sequential bag IDs (0, 1, 2, ...) is the simplest approach to avoid confusion.

## Sample code

```python
import numpy as np
from milgboost.datasets import make_mil_data
from milgboost.objective import LSEBCE
from milgboost.model import LightGBMMILModel

# Generate synthetic MIL data: 200 bags, 10 features
x, y, z = make_mil_data(
    n_bags=200,
    n_features=10,
    n_informative=5,
    key_instance_ratio=0.3,
    random_state=42,
)

# Split into train/test bags
n_train = 150
train_idx = z < n_train
test_idx = z >= n_train

x_train, y_train, z_train = x[train_idx], y[train_idx], z[train_idx]
x_test, y_test, z_test = x[test_idx], y[test_idx], z[test_idx]

# Train LSE-BCE LightGBM MIL model
model = LightGBMMILModel(
    objective=LSEBCE(r=1.0),
    lgb_params={"verbose": -1, "num_leaves": 15},
    num_boost_round=100,
)
model.fit(x_train, y_train, z_train)

# Predict
probs = model.predict_proba(x_test, z_test)
preds = model.predict(x_test, z_test)
print(f"Accuracy: {(preds == y_test[: len(preds)]).mean():.3f}")
```

## Development

```shell
git clone <repo>
cd milgboost

# Create virtualenv and install all extras + dev deps
uv sync --all-extras --group dev

# Type check
uv run poe check

# Lint & format
uv run poe lint
uv run poe format

# Run tests
uv run poe test
```

## License

MIT
