# Sentiment Analysis Project - Group 5

## Project Overview
This project implements a text classification system focusing on sentiment analysis. We explore and compare traditional machine learning models with deep learning models for sentiment classification.

## Team Members
- **Afsa Umutoniwase**
- **Benitha Uwituze Rutagengwa**
- **Jeremiah Olaitan Agbaje**
- **Tangue Kwizera**

## Project Structure
```
├── datasets/
│   └── raw/                      # Original dataset (IMDB, Twitter, Amazon reviews)
├── notebooks/
│   ├── sentiment_analysis_main.ipynb   # Main notebook: EDA, preprocessing, training, evaluation
│   └── experiments.ipynb               # Hyperparameter experiments and optimization
├── results/
│   └── figures/                  # Generated plots and visualizations
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```

## Dataset
**IMDB Dataset of 50K Movie Reviews**
- Source: [Kaggle - IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- File: `IMDB Dataset.csv`
- Size: 50,000 movie reviews
- Classes: Positive and Negative sentiment
- Features: Review text and sentiment labels

## Models Implemented
- **Traditional ML Model**: Logistic Regression with TF-IDF features
- **Deep Learning Model**: LSTM with word embeddings

## Installation and Setup
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the IMDB dataset from Kaggle and place `IMDB Dataset.csv` in `datasets/raw/`
4. Open and run `notebooks/sentiment_analysis_main.ipynb`

## Usage
1. Place your chosen dataset in `datasets/raw/`
2. Run `sentiment_analysis_main.ipynb` for the complete workflow
3. Use `experiments.ipynb` for hyperparameter optimization
4. Generated plots will be saved in `results/figures/`

## Results
[To be updated with model performance and key findings]

## References
[To be updated with proper citations in IEEE/APA format]
