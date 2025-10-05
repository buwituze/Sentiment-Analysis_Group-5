# Sentiment Analysis Project - Group 5

## Project Overview
This project implements a text classification system focusing on sentiment analysis. We explore and compare traditional machine learning models with deep learning models for sentiment classification.

## Team Members
## Project overview
This repository contains a sentiment analysis pipeline built and documented in `notebooks/sentiment_analysis_main.ipynb`. The work demonstrates end-to-end steps: exploratory data analysis (EDA), preprocessing, feature engineering (TF-IDF and Word2Vec), model training (traditional ML and a BiLSTM deep model), and evaluation on the IMDB movie reviews dataset.

## Team members
- Afsa Umutoniwase
- Benitha Uwituze Rutagengwa
- Jeremiah Olaitan Agbaje
- Tangue Kwizera

## Repository structure

```
. (repo root)
├── .git/                         # Git metadata
├── datasets/
│   └── raw/
│       └── IMDB Dataset.csv       # Original dataset (place here)
├── notebooks/
│   ├── sentiment_analysis_main.ipynb   # Main notebook: EDA, preprocessing, training, evaluation
│   └── ML(Naives_Bayes)_experiments.ipynb # Additional experiments
.├── README.md                     # Project documentation (this file)
.└── requirements.txt              # Python dependencies
```

## Dataset
- Name: IMDB Movie Reviews
- Source: Kaggle (place `IMDB Dataset.csv` inside `datasets/raw/`)
- Size: 50,000 reviews (25k positive, 25k negative)

Why this dataset: it's large and balanced, which is suitable for both classical ML and deep-learning experiments.

## Key findings from EDA (extracted from the main notebook)
- No missing values were found in the original dataset.
- Duplicate reviews: ~418 duplicates detected and removed.
- Review length varies widely (4 to ~2,470 words). After length-based filtering (10–500 words) the dataset was reduced by ~7.8%.
- Common positive words: great, excellent, best. Common negative words: bad, worst, boring.

## Preprocessing summary
- Length-based filtering: keep reviews with 10 to 500 words to remove very short or extremely long outliers.
- Text cleaning steps:
	- Lowercase conversion
	- Remove HTML tags (e.g., `<br />`)
	- Remove non-alphabetic characters (keep apostrophes for contractions)
	- Remove English stopwords
	- Remove numbers

- Effect: vocabulary size and average words per review reduced (reported ~46.6% word reduction in the notebook).

## Train / Validation / Test split
- Stratified split to preserve class balance:
	- Train: 32,899 samples (≈72%)
	- Validation: 3,656 samples (≈8%)
	- Test: 9,139 samples (≈20%)

## Feature engineering
- TF-IDF: 5,000 max features (sparse vectors) — effective for traditional ML models (Logistic Regression, SVM).
- Word2Vec embeddings: 100-dimensional vectors trained on the training set; per-review vector obtained by averaging word vectors — used with deep learning (BiLSTM).

## Models implemented
- Traditional ML: TF-IDF features with classical classifiers (e.g., Logistic Regression) — faster to train and interpretable.
- Deep Learning: Word2Vec-based embedding + Bidirectional LSTM (BiLSTM) with a small feed-forward head. The notebook trains with binary crossentropy and reports validation/test accuracy and classification reports.

## Results (notebook highlights)
- Cleaned datasets and feature matrices ready for modeling: TF-IDF (5,000 dims) and Word2Vec (100 dims).
- Splits maintained class balance across sets.
- Reported dataset sizes after preprocessing: 32,899 train / 3,656 val / 9,139 test.
- The notebook includes training curves, a confusion matrix, and precision/recall/f1 scores for the models.

### Model training results
The main BiLSTM (Word2Vec) run recorded in `notebooks/sentiment_analysis_main.ipynb` produced the following top-line metrics:

- Validation accuracy: 0.7458971553610503
- Test accuracy: 0.747346536820221

Interpretation: the BiLSTM model achieved ~74.6% accuracy on validation and ~74.7% on the held-out test set. Overall performance is consistent between validation and test, indicating the model generalizes reasonably well given the preprocessing and model configuration. The notebook contains the full classification reports (precision, recall, F1) and the confusion matrix for deeper inspection.

Full per-class classification metrics are available in the notebook outputs for reference.

## How to run
1. Create a Python environment and install dependencies:

```powershell
pip install -r requirements.txt
```

2. Download `IMDB Dataset.csv` from Kaggle and place it in `datasets/raw/`.
3. Open `notebooks/sentiment_analysis_main.ipynb` in Jupyter or Google Colab. The notebook includes data mounting instructions for Colab.
4. Run cells from top to bottom. Key configurable parameters in the notebook:
	 - Length filter: `MIN_WORDS`, `MAX_WORDS`
	 - TF-IDF `max_features`
	 - Word2Vec `vector_size`, `window`, `min_count`
	 - BiLSTM `MAX_VOCAB`, `MAX_LEN`, training `epochs` and `batch_size`

## Notes & next steps
- The notebook keeps preprocessing simple (no stemming/lemmatization) to retain word forms that may carry sentiment.
- Possible extensions:
	- Add experiments with n-grams or pretrained embeddings (GloVe/fastText)
	- Tune classical models (SVM, Logistic Regression) with grid search
	- Add inference scripts and model checkpoints under `models/`

## References
- IMDB Dataset of 50K Movie Reviews (Kaggle)
