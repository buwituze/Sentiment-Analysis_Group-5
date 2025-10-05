# Sentiment Analysis Project - Group 5

## Project ove## Data splits
We split the data to keep the same ratio of positive/negative reviews:
- Training: 32,899 samples (72%)
- Validation: 3,656 samples (8%)
- Test: 9,139 samples (20%)
We built a sentiment analysis system using movie reviews from IMDB. Our approach compares traditional machine learning (Naive Bayes) with deep learning (BiLSTM) to classify reviews as positive or negative. All our work is in the `notebooks/` folder.

## Team members
- Afsa Umutoniwase
- Benitha Uwituze Rutagengwa
- Jeremiah Olaitan Agbaje
- Tangue Kwizera

## Team contributions
- **Benitha Uwituze**: Data exploration, preprocessing pipeline, feature engineering (TF-IDF/Word2Vec), EDA visualizations, and preprocessing justifications
- **Jeremiah Olaitan**: Traditional ML model implementation (Naive Bayes), hyperparameter experimentation, alpha tuning analysis, and code documentation
- **Tangue Kwizera**: Deep learning model design (BiLSTM), architecture implementation, training pipeline setup, and results analysis
- **Afsa Umutoniwase**: Quality assurance, GitHub repository structure, evaluation metrics implementation, and project integration

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
├── README.md                     # Project documentation (this file)
└── requirements.txt              # Python dependencies
```

## Dataset
- IMDB Movie Reviews from Kaggle
- 50,000 reviews (25k positive, 25k negative)
- Place `IMDB Dataset.csv` in `datasets/raw/`

We chose this dataset because it's balanced and large enough for our experiments.

## What we found in the data
- No missing values
- Removed ~418 duplicate reviews
- Review lengths vary a lot (4 to 2,470 words), so we filtered to keep only 10-500 word reviews
- Positive reviews often use words like "great", "excellent", "best"
- Negative reviews use "bad", "worst", "boring"

## Data cleaning
We cleaned the text data by:
- Keeping only reviews with 10-500 words
- Converting to lowercase
- Removing HTML tags like `<br />`
- Removing punctuation (but keeping apostrophes)
- Removing stopwords and numbers

This reduced the vocabulary by about 46.6%.

## Train / Validation / Test split
- Stratified split to preserve class balance:
	- Train: 32,899 samples (≈72%)
	- Validation: 3,656 samples (≈8%)
	- Test: 9,139 samples (≈20%)

## Features
- **TF-IDF**: Converted text to 5,000 numerical features for the Naive Bayes model
- **Word2Vec**: Created 100-dimensional word vectors for the BiLSTM model

## Models we tried
- **Naive Bayes**: Used TF-IDF features, fast to train and easy to understand
- **BiLSTM**: Used Word2Vec embeddings with a bidirectional LSTM network

## Results
Here's what we got from our experiments:

### Model training results

#### Experiments from `notebooks/ML(Naives_Bayes)_experiments.ipynb`

### Model training results

#### Naive Bayes Alpha Experiments

We tested different alpha values to find the best smoothing parameter:

| Model | Alpha | Train Acc (%) | Val Acc (%) | Test Acc (%) | Gap | Notes |
|-------|-------|---------------|-------------|--------------|-----|-------|
| 1     | 0.01  | 87.14         | 85.20       | 85.75        | 1.39%| Low smoothing |
| 2     | 0.1   | 87.13         | 85.23       | 85.76        | 1.37%| Still low |
| 3     | 0.5   | 87.18         | 85.28       | 85.78        | 1.40%| Medium |
| 4     | 1.0   | 87.13         | 85.15       | 85.82        | 1.31%| Default |
| 5     | 2.0   | 87.11         | 85.42       | 85.94        | 1.17%| Getting better |
| 6     | 5.0   | 87.05         | 85.48       | 86.07        | 0.98%| **Best!** |
| 7     | 10.0  | 86.98         | 85.09       | 85.97        | 1.01%| Too much smoothing |

**What we learned:**
- Alpha = 5.0 gave us the best test accuracy (86.07%)
- Higher alpha values reduce overfitting
- Performance doesn't improve much after alpha = 5.0

#### Feature Engineering Results

| What we tested | Our choice | Performance | Could try next |
|----------------|------------|-------------|----------------|
| Vocabulary size | 5000 words | 86.07% | Try 10k or 15k |
| Text cleaning | Remove stopwords + lowercase | Good baseline | Add stemming |
| Feature type | Single words (TF-IDF) | Works well | Try word pairs |
| Smoothing | Alpha = 5.0 | Best balance | - |

**Takeaway:** Our current setup with 5000 TF-IDF features works well, but we could try more advanced preprocessing.

#### Deep Learning Results

**BiLSTM + Word2Vec:** 74.59% validation, 74.73% test accuracy

*Still need to try:* Different learning rates, batch sizes, and network architectures

**Summary:**
- Naive Bayes (86.07%) worked much better than BiLSTM (74.73%)
- Best alpha value for Naive Bayes was 5.0
- TF-IDF features were more effective than Word2Vec for this dataset

## How to run our code
1. Install the required packages:
```powershell
pip install -r requirements.txt
```

2. Download the IMDB dataset from Kaggle and put `IMDB Dataset.csv` in `datasets/raw/`

3. Open `notebooks/sentiment_analysis_main.ipynb` in Jupyter or Google Colab and run the cells from top to bottom

The main parameters you can change:
- Review length filter: `MIN_WORDS`, `MAX_WORDS`
- TF-IDF features: `max_features`
- Word2Vec settings: `vector_size`, `window`
- Training settings: `epochs`, `batch_size`

## What we could improve
- Try stemming or lemmatization
- Experiment with n-grams or pretrained embeddings
- Test other models like SVM or Logistic Regression
- Save trained models for future use

## References
- IMDB Dataset of 50K Movie Reviews (Kaggle)
