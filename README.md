# Vehicle Price Prediction

Predicting the listing price (USD) of recent-model-year US vehicles from their specifications, using a leak-free scikit-learn pipeline and a tuned Random Forest.

> **Final result (held-out test set, 190 vehicles):** R² = **0.884**, MAE = **$3,981**, MAPE = **8.39%** — a **73% lower error** than a naive "predict the average price" baseline ($14,799).

---

## Table of contents

1. [Project overview](#project-overview)
2. [Results](#results)
3. [Dataset](#dataset)
4. [Approach](#approach)
5. [Model selection and tuning](#model-selection-and-tuning)
6. [Error analysis](#error-analysis)
7. [Limitations](#limitations)
8. [Repository structure](#repository-structure)
9. [How to run](#how-to-run)
10. [Tech stack](#tech-stack)
11. [Author](#author)

---

## Project overview

The goal is to build a regression model that estimates a vehicle's price from fields such as make, model, year, engine description, transmission, trim, mileage and drivetrain.

The project walks through a complete, ordered machine-learning workflow:

1. Data cleaning and missing-value handling
2. Train/test split (the test set is kept untouched until the very end)
3. Feature engineering (parsing free-text `engine`, `transmission` and `trim` fields)
4. Encoding and scaling
5. Baseline models and cross-validation
6. Packaging all preprocessing into a single `Pipeline`
7. Hyperparameter tuning and controlled experiments
8. Error analysis
9. One final evaluation on the test set

## Results

### Final model (evaluated once on the test set)

| Metric | Train | Test |
| --- | --- | --- |
| R² | 0.969 | **0.884** |
| MAE | $1,844 | **$3,981** |
| MAPE | – | 8.39% |

| Baseline | MAE on test |
| --- | --- |
| Dummy (always predicts the mean train price) | $14,799 |
| **Final model** | **$3,981** (−73.10%) |

### Cross-validation estimate (5-fold, training data only)

| Metric | Train folds | Validation folds |
| --- | --- | --- |
| R² | 0.967 | 0.778 (± 0.075) |
| MAE | $1,920 | $4,811 (± $599) |

The test score is higher than the cross-validation average. With only 190 test vehicles and a fold-to-fold standard deviation of ±0.075 in R², a realistic expectation for new vehicles is **R² ≈ 0.78–0.88** rather than a single exact number. The gap between train and validation is expected for a Random Forest, and the experiments below showed that reducing model complexity made validation performance worse, not better.

## Dataset

- ~1,000 new-vehicle listings (1,002 raw rows), recent model years.
- Original columns: `name`, `description`, `make`, `model`, `year`, `price`, `engine`, `cylinders`, `fuel`, `mileage`, `transmission`, `trim`, `body`, `doors`, `exterior_color`, `interior_color`, `drivetrain`.
- Target: `price` (USD).

### Cleaning steps

| Step | Rows remaining |
| --- | --- |
| Raw data | 1,002 |
| Dropped 24 duplicate rows | 978 |
| Dropped 1 row with `price = 0` | 977 |
| Dropped 23 rows with missing `price` (target) | 954 |
| Dropped rows missing `transmission`, `trim`, `body` or `engine` | **946** |
| Train / test split (80 / 20, `random_state=32`) | **756 / 190** |

Other missing values: `mileage` → median; `fuel`, `doors` → mode; `cylinders` → most common value for the same `make`.
Dropped columns: `name`, `description`, `exterior_color`, `interior_color` (the free-text `description` is left for a future NLP step).

## Approach

All feature engineering and encoding live inside one scikit-learn `Pipeline`, so every step is **fit on training data only** and then applied unchanged to validation folds and the test set.

```
raw data
  → ExtractEngineFeatures        (custom transformer)
  → ExtractTransmissionFeatures  (custom transformer)
  → GroupRareCategories(model, top 25)
  → GroupRareCategories(trim,  top 50)
  → ColumnTransformer            (one-hot / ordinal / robust scaling)
  → RandomForestRegressor
```

### Feature engineering

| Source column | Engineered features |
| --- | --- |
| `engine` (free text) | 14 binary flags: V6, V8, I4, turbo, hybrid, diesel, electric, 12V / 16V / 24V valves, DOHC, SOHC, OHV, VVT |
| `transmission` (free text) | `gears` (number of speeds, missing values filled with the training median), `is_cvt`, `is_dual_clutch`, `has_manual_mode` |
| `model` | top 25 most frequent models kept, the rest grouped as `other` |
| `trim` | top 50 most frequent trims kept, the rest grouped as `other` |

The transmission column originally had 29 distinct spellings of only a handful of real types, and the engine column was similar. Parsing them into features removed that noise.

### Encoding and scaling

| Columns | Method |
| --- | --- |
| `drivetrain`, grouped `model`, grouped `trim` | One-hot encoding (unknown categories ignored) |
| `make`, `fuel`, `body` | Ordinal encoding (unknown categories → −1) |
| `year`, `cylinders`, `mileage`, `doors`, `gears` | `RobustScaler` (robust to the extreme mileage outliers) |
| engine / transmission flags | Passed through unchanged |

The preprocessing produces **86 features**.

## Model selection and tuning

### Baselines (5-fold CV on training data)

| Model | R² | MAE |
| --- | --- | --- |
| Dummy (mean) | -0.004 | $13,856 |
| Linear Regression | 0.590 | $7,403 |
| Decision Tree | 0.640 | $5,637 |
| Random Forest (default) | 0.735 | $5,186 |
| Random Forest in the full pipeline | 0.736 | $5,136 |

Random Forest was chosen for the next steps. Putting everything into a pipeline gave almost the same score, confirming the manual preprocessing and the pipeline were equivalent.

### Experiments (one change at a time, each compared with 5-fold CV)

| Experiment | Result | Decision |
| --- | --- | --- |
| Gradient Boosting / HistGradientBoosting | R² 0.678 / 0.740, MAE $6,149 / $5,968 | Not better than Random Forest |
| Randomized search on Random Forest (20 combinations) | Best: 600 trees, `max_features=0.3` → MAE $4,975, R² 0.767 | **Adopted** (`max_features` was the setting that mattered) |
| Grid search on `top_k` for `trim` and `model` | `trim` 50 / `model` 25 → MAE ≈ $4,811 | **Adopted** (keeping more trims clearly helped) |
| Feature selection (`SelectFromModel`) | MAE $5,003 vs $4,811, worse in 5 of 5 folds | Rejected |
| Reducing model complexity (`min_samples_leaf`, `max_depth`) | Validation MAE $4,933–$6,506 | Rejected |
| `log1p` transform of the target | MAE $4,735 vs $4,811, better in only 3 of 5 folds | Rejected (decision rule fixed before running: at least 4 of 5 folds) |

Decisions were taken on cross-validation results only. The test set was used **once**, at the end.

## Error analysis

Using out-of-fold predictions on the training data:

| Price quartile | Mean absolute error | Mean error (% of price) |
| --- | --- | --- |
| Cheapest 25% | $3,200 | 10.7% |
| 25–50% | $3,231 | 7.6% |
| 50–75% | $3,642 | 6.9% |
| **Most expensive 25%** | **$9,184** | 11.0% |

The largest errors are rare, very expensive vehicles (for example high-end electric and full-size luxury SUVs) that the model under-estimates. The percentage error is fairly consistent across price ranges, but the dollar error is dominated by the top quartile. The top 10 errors came from realistic prices, not data-entry mistakes.

## Limitations

- **Small dataset** (946 vehicles after cleaning), so results vary noticeably between folds and between splits. The test set (190 vehicles) is a single, noisy estimate.
- **Some imputation happened before the train/test split** (median/mode fills for `mileage`, `fuel`, `cylinders`, `doors`). The effect is likely small, but a stricter setup would move these into the pipeline.
- The model is weaker on **rare, very high-priced vehicles**.
- The data is a single snapshot of **new** vehicle listings (mileage is mostly near zero), so the model should not be expected to generalise to used cars, older vehicles or other markets.
- `description` and color columns were not used.

## Repository structure

```
.
├── README.md
├── requirements.txt
└── vehicle-price-prediction.ipynb   # full workflow: cleaning → pipeline → tuning → final test
```

## How to run

1. Clone the repository and create a virtual environment:

   ```bash
   git clone https://github.com/mostafa3twa/vehicle-price-prediction.git
   cd vehicle-price-prediction
   python -m venv .venv
   source .venv/bin/activate        # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Open the notebook and point the first data-loading cell to your own copy of the data file (the notebook was written on Kaggle, so the original path will not exist on your machine):

   ```python
   df = pd.read_csv("path/to/your/data.csv")
   ```

3. Run the notebook from top to bottom:

   ```bash
   jupyter notebook vehicle-price-prediction.ipynb
   ```

The data file is not included in this repository. The notebook is saved with all its outputs, so every result in this README can be read directly on GitHub without re-running anything.

The notebook was developed with Python 3.13 and scikit-learn 1.6.1. Using the same scikit-learn version is recommended.

## Tech stack

Python · pandas · NumPy · scikit-learn (`Pipeline`, `ColumnTransformer`, custom transformers, `RandomizedSearchCV`, `GridSearchCV`) · matplotlib · seaborn · Jupyter

## Author

Mostafa — [github.com/mostafa3twa](https://github.com/mostafa3twa)
