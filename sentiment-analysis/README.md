# Sentiment Analysis

A scikit-learn based sentiment analysis notebook that classifies movie reviews
as positive (`1`) or negative (`0`).

## Files

- `sentiment_1.ipynb` — end-to-end notebook: loads the corpus, builds
  `movie_reviews.csv`, preprocesses text, and trains `LogisticRegression` and
  `MultinomialNB` models with TF-IDF features.
- `movie_reviews.csv` — the prepared dataset (already generated and committed).
- `sentiment-analysis/requirements.txt` — Python dependencies.

## Requirements

- Python 3
- pandas, scikit-learn, nltk

Install dependencies:

```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

## Usage

Open the notebook in Jupyter and run the cells top-to-bottom:

```bash
jupyter notebook sentiment_1.ipynb
```

### Notes

- `movie_reviews.csv` is already included, so you do **not** need the raw text
  corpus. The early cells that build the CSV assume a local corpus under a
  hardcoded path (`D:\pythonducat\...`) and are only needed if you want to
  regenerate the dataset from raw review files.
- The pipeline uses scikit-learn's built-in `stop_words='english'`, so the
  `nltk` stopwords download is only required if you run the NLTK cells
  directly.
