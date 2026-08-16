# Weather Forecasting

Predicts weather conditions from historical Delhi weather observations using
scikit-learn (`KNeighborsClassifier` and `RandomForestClassifier`).

## Files

- `weather_prediction.ipynb` — data cleaning, preprocessing, training, and
  evaluation notebook.
- `testset.csv` — the Delhi weather dataset used by the notebook (extracted
  from the original `delhi-weather-data.zip`).
- `weather-forecasting/requirements.txt` — Python dependencies.

## Requirements

- Python 3
- numpy, pandas, scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Extract the dataset (already done — `testset.csv` is included) and run the
notebook:

```bash
jupyter notebook weather_prediction.ipynb
```

The notebook reads `testset.csv` from this folder, cleans the data, label-encodes
the condition column, and evaluates KNN and Random Forest classifiers.
