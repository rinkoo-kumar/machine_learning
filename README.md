# machine_learning

A collection of machine learning projects built with Python and
[scikit-learn](https://scikit-learn.org/).

## Projects

- [handwritten-digit-recognition](handwritten-digit-recognition) — MNIST digit
  recognition using `LogisticRegression`, `RandomForest`, and
  `GradientBoosting` (with `StandardScaler` + `PCA`) plus a Tkinter GUI.
- [sentiment-analysis](sentiment-analysis) — sentiment classification on movie
  reviews using scikit-learn.

## Structure

```
machine_learning/
├── handwritten-digit-recognition/
│   ├── train.py          # trains and saves the model pipeline
│   ├── digit_gui.py      # Tkinter prediction GUI
│   ├── requirements.txt
│   └── README.md
└── sentiment-analysis/
    ├── sentiment_1.ipynb
    ├── movie_reviews.csv
    └── README.md
```
