# MLX Solar Radiation Prediction

This project solves a solar radiation forecasting task: predict `Radiation` in W/m² from historical atmospheric and environmental sensor data. The solution is designed for Kaggle-style hackathon execution, where the notebook trains inside Kaggle and writes a final `submission.csv`.

## Problem Solved

Solar radiation is strongly affected by time of day, day of year, sun angle, cloud/moisture conditions, and recent weather behavior. A naive model can collapse into a few repeated values, especially around night/day boundaries. This project addresses that by combining:

- leakage-safe chronological processing
- local solar-time feature engineering
- clear-sky solar physics
- atmospheric moisture physics
- shifted time-series memory features
- gradient boosted tree ensembles
- Ridge stacking
- submission diagnostics to catch prediction collapse

The final notebook is self-contained and can be run top to bottom in Kaggle.

## Main File

```text
solar_stack_pipeline.ipynb
```

This notebook includes the complete training and inference pipeline:

- package installation
- automatic Kaggle input file discovery
- data loading and sorting
- feature engineering
- model training
- hyperparameter tuning
- stacking
- prediction clipping
- final submission generation
- sanity diagnostics

## Data Files Expected

The notebook expects these files to exist somewhere under `/kaggle/input`:

```text
train_df_1.csv
test_df_1.csv
```

In Kaggle, upload the notebook and add a dataset containing those two CSV files.

## Approach

### 1. Leakage-Safe Data Processing

The training and test data are sorted by `UNIXTime` before feature generation. This preserves the time-series order and prevents future-looking validation mistakes.

### 2. Local Time Correction

Solar features are calculated from the dataset's local `Time` and `Data` columns instead of directly using `UNIXTime % 86400`. This matters because sun position must align with local solar time, not a mismatched timestamp offset.

### 3. Solar Physics Features

The notebook computes a clear-sky radiation baseline using latitude `22.5° N`:

```text
cos_zenith
clear_sky_rad
solar_zenith_angle
```

These features tell the model the theoretical maximum radiation at each timestamp.

### 4. Atmospheric Physics Features

The notebook calculates dew point and dew point gap using the Magnus-Tetens relationship. These features help represent moisture, fog, and cloud formation risk.

```text
dew_point
dew_point_gap
```

### 5. Shifted Time-Series Features

The pipeline creates shifted lag and rolling features for weather sensors and known historical radiation. All time-series features use shifted values so the current row does not see its own target.

Examples:

```text
Temperature_lag_1
Humidity_lag_1
Pressure_lag_1
known_radiation_lag_1
known_radiation_rmean_3
known_radiation_rstd_6
```

### 6. Model Architecture

The final model is a stacked ensemble:

```text
Level 0: LightGBM, XGBoost, CatBoost
Level 1: Ridge Regression
```

Validation uses `TimeSeriesSplit`, not random K-fold cross-validation.

### 7. Target Transformation

The target is transformed with:

```python
np.log1p(Radiation)
```

Negative sensor noise is clipped to zero before transformation.

### 8. Physical Post-Processing

Predictions are transformed back using `expm1` and clipped to physical bounds:

```text
minimum: 0
maximum: clear_sky_rad
```

This prevents negative radiation and avoids artificial prediction spikes.

## Kaggle Usage

Run the notebook top to bottom in Kaggle. By default, it uses:

```python
RUN_MODE = "full"
```

The notebook auto-detects:

```text
train_df_1.csv
test_df_1.csv
```

and writes:

```text
/kaggle/working/submission.csv
/kaggle/working/features_used.json
```

If you need a faster check before the final run, change:

```python
RUN_MODE = "quick"
```

For a medium run:

```python
RUN_MODE = "tuned"
```

For the final run:

```python
RUN_MODE = "full"
```

## Submission Diagnostics

The notebook prints checks such as:

```text
Exact 50.0 count
Zero fraction
Night fraction
Zeros in 8am-4pm
Target standard deviation
Target maximum
```

These checks are included to catch model collapse before submission.

## Recommended GitHub Repository Contents

Commit these files:

```text
README.md
LICENSE
requirements.txt
.gitignore
solar_stack_pipeline.ipynb
```

Do not commit generated artifacts unless required by the hackathon:

```text
submission.csv
submission_100trial.csv
submission_v2.csv
features_used.json
features_100trial.json
tuning_100trial.out
```

Do not commit competition data unless the rules explicitly allow it:

```text
train_df_1.csv
test_df_1.csv
```

For Kaggle execution, upload `train_df_1.csv` and `test_df_1.csv` as Kaggle input data instead.

## Local Usage

If running outside Kaggle, place the CSV files next to the notebook and run all cells. The notebook falls back to local paths when `/kaggle/input` is unavailable.

## Requirements

```text
numpy
pandas
scikit-learn
lightgbm
xgboost
catboost
optuna
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
