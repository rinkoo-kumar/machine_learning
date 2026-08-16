# machine_learning

A Python digit recognition project built during a data science and machine
learning.

## Overview

This project trains several classifiers on the MNIST handwritten-digit dataset
and provides a simple Tkinter GUI to load an image (`img.png`) and predict the
digit using a majority vote across three models.

## Files

- `train.py` — trains `LogisticRegression`, `RandomForestClassifier`,
  and `GradientBoostingClassifier` on MNIST after `StandardScaler` + `PCA`
  preprocessing, then pickles the pipeline to `training.pkl`.
- `digit_gui.py` — Tkinter app (`Digit Recognition`) that loads `img.png`,
  converts it to grayscale, resizes to 28x28, applies the saved pipeline, and
  shows the majority-voted prediction.
- `training.pkl` — pickled model pipeline (scaler, PCA, and the three models).
- `metrics.json` — validation accuracies of the three models from the last
  training run.
- `img.png` — input image used by the GUI for prediction (must be present).

## Requirements

- Python 3
- numpy
- scikit-learn
- opencv-python (`cv2`)
- tkinter (usually bundled with Python)

Install dependencies:

```bash
pip install numpy scikit-learn opencv-python
```

## Usage

1. Train and save the models:

```bash
python train.py
```

2. Run the GUI and click **Load** then **predict**:

```bash
python digit_gui.py
```

## Metrics

`train.py` evaluates each model on the held-out MNIST test split and writes the
accuracies to `metrics.json`. In the GUI, click **metrics** to view the saved
validation accuracies.
