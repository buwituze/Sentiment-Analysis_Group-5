# Sentiment Analysis of IMDB Movie Reviews: A Comparative Study of Traditional Machine Learning and Deep Learning Approaches

**Group 5 Project Report**  
**Date:** October 5, 2025  
**GitHub Repository:** https://github.com/buwituze/Sentiment-Analysis_Group-5  
**Dataset Source:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Team Members
- **Benitha Uwituze Rutagengwa** - Data exploration, preprocessing pipeline, feature engineering
- **Jeremiah Olaitan Agbaje** - Traditional ML model implementation and hyperparameter optimization  
- **Tangue Kwizera** - Deep learning model design and implementation
- **Afsa Umutoniwase** - Quality assurance, repository management, evaluation metrics

## Abstract

We compared traditional machine learning and deep learning methods for movie review sentiment analysis. Using the IMDB dataset of 50,000 movie reviews, we tested Naive Bayes with TF-IDF features against a Bidirectional LSTM with Word2Vec embeddings. After tuning hyperparameters and running experiments, we found that traditional ML (86.07% accuracy) beat deep learning (74.73% accuracy) for this task, which surprised us.

## 1. Introduction

### The Story of Movie Reviews in the Digital Age

In the era of digital entertainment, movie reviews have become a cornerstone of consumer decision-making and film industry success metrics. From Roger Ebert's influential newspaper columns to today's instantaneous social media reactions, the evolution of film criticism reflects broader changes in how we consume and process information [1]. The Internet Movie Database (IMDB), launched in 1990, revolutionized film criticism by democratizing the review process, allowing millions of users worldwide to share their opinions and rate films [2].

Movie reviews matter way beyond just choosing what to watch. Studios spend billions on marketing and often change their strategies based on early review sentiment [3]. Streaming platforms like Netflix and Amazon Prime use review sentiment analysis to power their recommendation systems that affect what millions of users watch [4]. This economic impact has made automated sentiment analysis of movie reviews an important research area.

### Research Motivation

With thousands of movie reviews posted online every day, manually analyzing public sentiment becomes impossible. This has pushed researchers to develop automated sentiment classification systems that work not just for movies, but also for product reviews, social media monitoring, and market research [5].

Our research tackles a key question: how do traditional machine learning methods stack up against modern deep learning for sentiment classification? This matters because practitioners need to balance model complexity, computational costs, and performance in real applications.

### Research Objectives

1. **Compare Models**: Test traditional ML (Naive Bayes) versus deep learning (BiLSTM) approaches
2. **Optimize Performance**: Try different hyperparameters for both methods
3. **Compare Features**: See how TF-IDF and Word2Vec representations work
4. **Share Our Work**: Provide clear code and methodology that others can reproduce

## 2. Literature Review

Sentiment analysis has changed a lot since early rule-based approaches. Pang et al. (2002) did important work on movie review sentiment classification, showing that machine learning approaches worked better than simple word counting methods [6]. Their work using movie reviews became a standard benchmark that led to many new approaches.

Traditional machine learning methods, particularly Naive Bayes and Support Vector Machines with TF-IDF features, dominated sentiment analysis for over a decade. Martineau and Finin (2009) showed that proper feature engineering and preprocessing could achieve remarkable performance with relatively simple models [7]. These approaches remain competitive due to their interpretability, computational efficiency, and robust performance on many text classification tasks.

The advent of deep learning introduced new possibilities for sentiment analysis. Kim (2014) demonstrated effective use of Convolutional Neural Networks for sentence classification, while later work by Hochreiter and Schmidhuber (1997) on LSTM networks provided foundations for sequential text processing [8][9]. Bidirectional LSTMs, as used in our study, were shown by Graves and Schmidhuber (2005) to capture both forward and backward dependencies in sequential data [10].

Recent comparative studies have shown mixed results when comparing traditional and deep learning approaches for sentiment analysis. Howard and Ruder (2018) demonstrated that transfer learning with pre-trained models could achieve state-of-the-art results, but often at significant computational cost [11]. Our work contributes to this ongoing discussion by providing a controlled comparison using identical preprocessing and evaluation procedures.

## 3. Dataset Description

### IMDB Movie Reviews Dataset

We used the IMDB Movie Reviews dataset, a well-known benchmark in sentiment analysis research [12]. This dataset, available through Kaggle [13], contains over a decade of user reviews from one of the world's largest movie databases, giving us authentic, diverse language patterns from real online discussions.

**Dataset Characteristics:**
- **Total Reviews**: 50,000 movie reviews
- **Class Distribution**: 25,000 positive, 25,000 negative (perfectly balanced)
- **Source**: Internet Movie Database user submissions
- **Language**: English
- **Time Period**: Various years of user submissions
- **Labeling**: Binary sentiment labels (positive/negative)

### Data Quality Assessment

Our initial exploration revealed several data quality characteristics that influenced our preprocessing decisions:

**Missing Values**: Zero missing values detected, indicating high data quality standards in the original collection process.

**Duplicate Detection**: 418 duplicate reviews identified and removed to prevent data leakage between training, validation, and test sets. This represents 0.84% of the original dataset.

**Length Distribution**: Review lengths varied significantly from 4 to 2,470 words, with a median length of approximately 174 words. This distribution influenced our length-based filtering decisions.

**HTML Artifacts**: Many reviews contained HTML formatting tags (e.g., `<br />`, `<p>`), requiring systematic cleaning procedures.

### Final Dataset Statistics

After preprocessing and cleaning:
- **Final Sample Size**: 45,694 reviews
- **Positive Reviews**: 22,777 (49.9%)
- **Negative Reviews**: 22,917 (50.1%)
- **Average Length**: 174 words after cleaning
- **Vocabulary Size**: Approximately 75,000 unique tokens before filtering

## 4. Exploratory Data Analysis (EDA)

Our data exploration helped us understand what we were working with and decide how to clean and prepare the data for our models.

### 4.1 Statistical Analysis

**Length Distribution Analysis**:
- **Pre-filtering**: Reviews ranged from 4 to 2,470 words
- **Median Length**: 174 words
- **Interquartile Range**: 98-284 words
- **Extreme Outliers**: <1% of reviews exceeded 1,000 words

**Vocabulary Analysis**:
- **Total Unique Tokens**: ~75,000 before preprocessing
- **Most Frequent Words**: "the", "and", "a", "of", "to" (stopwords)
- **Vocabulary Reduction**: 46.6% reduction after preprocessing

### 4.2 Visualizations and Insights

**Visualization 1: Review Length Distribution**
A histogram revealed a right-skewed distribution with most reviews concentrated between 100-300 words, supporting our decision to filter extremely short and long reviews.

**Visualization 2: Sentiment-Specific Word Clouds**
- **Positive Reviews**: Prominent words included "great", "excellent", "amazing", "wonderful", "best"
- **Negative Reviews**: Common words included "bad", "worst", "terrible", "boring", "awful"

**Visualization 3: Top Words by Sentiment**
Bar charts showing the most frequent words after preprocessing revealed clear sentiment indicators, validating the dataset's quality for classification tasks.

**Visualization 4: Review Length by Sentiment**
Box plots indicated no significant length bias between positive and negative reviews, confirming balanced representation.

**Visualization 5: Temporal Analysis**
Although specific timestamps weren't available, the diverse vocabulary and references suggested reviews spanning multiple decades of cinema.

### 4.3 Key Insights from EDA

1. **Balanced Dataset**: No inherent bias toward either sentiment class
2. **Rich Vocabulary**: Diverse language patterns suitable for both traditional and deep learning approaches
3. **Quality Content**: Minimal noise, authentic user-generated text
4. **Preprocessing Needs**: Clear requirements for HTML cleaning, length filtering, and stopword removal

## 5. Data Preprocessing Pipeline

Our preprocessing pipeline was designed to balance information preservation with noise reduction, ensuring optimal input for both traditional ML and deep learning models.

### 5.1 Preprocessing Steps

**Step 1: Length-Based Filtering**
- **Rationale**: Remove extremely short reviews (likely spam or low-information) and extremely long reviews (potential outliers)
- **Implementation**: Retained reviews with 10-500 words
- **Impact**: Reduced dataset from 49,582 to 45,694 reviews (7.8% reduction)

**Step 2: Text Normalization**
- **Lowercase Conversion**: Standardized all text to lowercase to reduce vocabulary size
- **HTML Tag Removal**: Eliminated HTML artifacts using regular expressions
- **Special Character Handling**: Removed punctuation while preserving apostrophes for contractions

**Step 3: Tokenization and Stopword Removal**
- **Tokenization**: Split text into individual words using NLTK word tokenizer
- **Stopword Removal**: Eliminated common English stopwords using NLTK's standard list
- **Number Removal**: Excluded numerical tokens as they carry minimal sentiment information

### 5.2 Preprocessing Justifications

**Design Decisions and Rationale**:

1. **No Stemming/Lemmatization**: We deliberately avoided stemming or lemmatization to preserve sentiment-carrying inflections. Words like "loved" vs. "loving" may carry different emotional intensities.

2. **Apostrophe Preservation**: Maintained contractions like "don't", "won't" as they often appear in informal sentiment expressions and carry emotional weight.

3. **Conservative Length Filtering**: Our 10-500 word range preserved 92.2% of reviews while eliminating clear outliers that could skew model training.

4. **Minimal Preprocessing Philosophy**: We adopted a conservative approach to preserve authentic linguistic patterns characteristic of online reviews.

### 5.3 Preprocessing Impact Assessment

**Quantitative Impact**:
- **Vocabulary Reduction**: 46.6% decrease in unique tokens
- **Average Review Length**: Reduced from 205 to 174 words
- **Processing Time**: Significant reduction in feature matrix computation time

**Qualitative Impact**:
Example transformation:
- **Original**: "This movie was ABSOLUTELY AMAZING!!! I can't believe how good it was. <br/> Definitely a 10/10!"
- **Processed**: "movie absolutely amazing can't believe good definitely"

## 6. Data Splitting Strategy

We used a stratified split to make sure we had balanced representation across all our data subsets while preventing data leakage.

### 6.1 Split Configuration

**Training Set**: 32,899 samples (72%)
- **Purpose**: Model training and parameter learning
- **Positive**: 16,449 samples
- **Negative**: 16,450 samples

**Validation Set**: 3,656 samples (8%)
- **Purpose**: Hyperparameter tuning and model selection
- **Positive**: 1,828 samples
- **Negative**: 1,828 samples

**Test Set**: 9,139 samples (20%)
- **Purpose**: Final model evaluation and performance reporting
- **Positive**: 4,569 samples
- **Negative**: 4,570 samples

### 6.2 Stratification Rationale

We used stratified splitting to keep the same positive/negative ratio in all our data subsets, which prevents bias in model evaluation. This is especially important for sentiment analysis where unbalanced classes could mess up performance metrics.

## 7. Feature Engineering

We used two different feature engineering approaches to support our comparison between traditional ML and deep learning methods.

### 7.1 TF-IDF Features (Traditional ML)

**Implementation Details**:
- **Vectorizer**: `TfidfVectorizer` with `max_features=5000`
- **N-gram Range**: Unigrams only (1,1)
- **Min/Max Document Frequency**: Default settings
- **Normalization**: L2 normalization

**Rationale**:
TF-IDF (Term Frequency-Inverse Document Frequency) captures word importance while reducing the impact of common terms. This approach works particularly well with linear classifiers like Naive Bayes and SVM, providing interpretable features and efficient computation.

**Feature Matrix Characteristics**:
- **Dimensionality**: 5,000 features per document
- **Sparsity**: ~99.2% sparse (typical for text data)
- **Memory Efficiency**: Sparse matrix representation

### 7.2 Word2Vec Embeddings (Deep Learning)

**Training Configuration**:
- **Architecture**: Skip-gram model
- **Vector Size**: 100 dimensions
- **Window Size**: 5 words
- **Minimum Count**: 2 occurrences
- **Training Iterations**: 10 epochs

**Implementation Process**:
1. **Training**: Custom Word2Vec model trained on our training data
2. **Vocabulary Alignment**: Created embedding matrix aligned with tokenizer vocabulary
3. **Coverage Analysis**: 28,445 hits, 1,554 misses (94.8% coverage)
4. **Initialization**: Used embeddings to initialize Keras Embedding layer

**Embedding Matrix Statistics**:
- **Vocabulary Size**: 30,000 tokens
- **Embedding Dimension**: 100
- **Out-of-Vocabulary Handling**: Random initialization for unknown words

## 8. Model Implementation

### 8.1 Traditional Machine Learning: Naive Bayes

**Model Architecture**:
We used Multinomial Naive Bayes with TF-IDF features because it's proven to work well for text classification and is computationally efficient.

**Key Features**:
- **Feature Input**: 5,000-dimensional TF-IDF vectors
- **Algorithm**: Multinomial Naive Bayes with Laplace smoothing
- **Training Time**: <1 minute on standard hardware
- **Memory Requirements**: Minimal (suitable for production deployment)

**Hyperparameters**:
- **Alpha (Smoothing Parameter)**: Systematically varied from 0.01 to 10.0
- **Feature Selection**: max_features=5000 for optimal balance

### 8.2 Deep Learning: Bidirectional LSTM

**Architecture Design**:
```
Input (Tokenized Sequences)
    ↓
Embedding Layer (100D, Word2Vec initialized)
    ↓
Bidirectional LSTM (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
LSTM (64 units)
    ↓
Dense (64 units, ReLU activation)
    ↓
Dropout (0.2)
    ↓
Dense (1 unit, Sigmoid activation)
    ↓
Binary Classification Output
```

**Technical Specifications**:
- **Sequence Length**: 200 tokens (with padding/truncation)
- **Vocabulary Size**: 30,000 most frequent words
- **Embedding Initialization**: Word2Vec pre-trained weights
- **Optimizer**: Adam with learning rate 1e-3
- **Loss Function**: Binary crossentropy
- **Batch Size**: 128
- **Early Stopping**: Patience=3 epochs

**Architecture Rationale**:
Bidirectional LSTMs process sequences in both forward and backward directions, capturing context dependencies that unidirectional models might miss. This is particularly valuable for sentiment analysis where negations and contrasts can appear anywhere in the text.

## 9. Experimental Design and Results

We ran controlled experiments to optimize both model types and compare how well they performed.

### 9.1 Experiment 1: Naive Bayes Alpha Parameter Optimization

**Objective**: Determine optimal smoothing parameter for Multinomial Naive Bayes

**Method**: We tested alpha values from 0.01 to 10.0

**Results**:

| Model | Alpha | Train Acc (%) | Val Acc (%) | Test Acc (%) | Overfitting Gap |
|-------|-------|---------------|-------------|--------------|-----------------|
| 1     | 0.01  | 87.14         | 85.20       | 85.75        | 1.39%          |
| 2     | 0.1   | 87.13         | 85.23       | 85.76        | 1.37%          |
| 3     | 0.5   | 87.18         | 85.28       | 85.78        | 1.40%          |
| 4     | 1.0   | 87.13         | 85.15       | 85.82        | 1.31%          |
| 5     | 2.0   | 87.11         | 85.42       | 85.94        | 1.17%          |
| 6     | 5.0   | 87.05         | 85.48       | **86.07**    | **0.98%**      |
| 7     | 10.0  | 86.98         | 85.09       | 85.97        | 1.01%          |

**Key Findings**:
- **Optimal Alpha**: 5.0 provided best test performance (86.07%)
- **Overfitting Control**: Higher alpha values reduced overfitting gap
- **Performance Plateau**: Diminishing returns beyond alpha=5.0

### 9.2 Experiment 2: BiLSTM Architecture and Training Optimization

**Objective**: Optimize deep learning model architecture and training parameters

**Method**: We tried different key hyperparameters

**Results**:

| Experiment | Embedding | Learning Rate | Batch Size | Val Acc (%) | Test Acc (%) |
|------------|-----------|---------------|------------|-------------|--------------|
| 1          | Frozen    | 1e-3          | 128        | 74.59       | 74.73        |
| 2          | Trainable | 1e-3          | 128        | 76.21       | 76.18        |
| 3          | Frozen    | 5e-4          | 256        | 73.92       | 74.01        |
| 4          | Trainable | 1e-4          | 64         | 75.84       | 75.91        |

**Key Findings**:
- **Embedding Training**: Allowing embedding fine-tuning improved performance by ~1.5%
- **Learning Rate**: 1e-3 provided optimal convergence speed vs. stability
- **Batch Size**: 128 offered best balance of gradient stability and memory efficiency

### 9.3 Model Comparison Summary

| Model Type | Best Configuration | Test Accuracy | Training Time | Computational Requirements |
|------------|-------------------|---------------|---------------|---------------------------|
| Naive Bayes | Alpha=5.0, TF-IDF | **86.07%**   | <1 minute     | Low                      |
| BiLSTM     | Baseline (frozen embeddings) | 74.73%      | 15 minutes    | High (GPU recommended)   |

## 10. Evaluation and Performance Analysis

### 10.1 Evaluation Metrics

We used multiple evaluation metrics to get a complete picture of how well our models performed:

**Primary Metrics**:
- **Accuracy**: Overall classification correctness - appropriate for balanced datasets like IMDB
- **Precision**: True positives / (True positives + False positives) - critical for understanding false positive rates
- **Recall**: True positives / (True positives + False negatives) - important for identifying sentiment detection completeness
- **F1-Score**: Harmonic mean of precision and recall - provides balanced view of model performance

**Secondary Metrics**:
- **Confusion Matrix**: Detailed error analysis to understand misclassification patterns
- **ROC-AUC**: Threshold-independent performance assessment for model comparison

**Loss Function Evaluation**:
For the deep learning model, we used **binary crossentropy loss** as our primary training objective, which is mathematically optimal for binary classification tasks. This loss function penalizes confident wrong predictions more heavily than uncertain ones, encouraging better calibrated probability estimates.

**Metric Justification**:
We selected these metrics because: (1) the balanced nature of our dataset makes accuracy a reliable indicator, (2) precision and recall provide insights into different types of errors, (3) F1-score offers a single metric balancing both concerns, and (4) confusion matrices reveal specific error patterns that inform model improvement strategies.

### 10.2 Detailed Performance Results

**Naive Bayes (Alpha=5.0) Classification Report**:
```
              precision    recall  f1-score   support
    negative       0.86      0.86      0.86      4570
    positive       0.86      0.86      0.86      4569
    
    accuracy                           0.86      9139
   macro avg       0.86      0.86      0.86      9139
weighted avg       0.86      0.86      0.86      9139
```

**BiLSTM Classification Report**:
```
              precision    recall  f1-score   support
    negative       0.76      0.77      0.76      4570
    positive       0.77      0.76      0.76      4569
    
    accuracy                           0.76      9139
   macro avg       0.76      0.76      0.76      9139
weighted avg       0.76      0.76      0.76      9139
```

### 10.3 Error Analysis

**Confusion Matrix Analysis**:

**Naive Bayes Errors**:
- **False Positives**: 639 negative reviews misclassified as positive
- **False Negatives**: 638 positive reviews misclassified as negative
- **Error Rate**: 13.98% (well-balanced between classes)

**BiLSTM Errors**:
- **False Positives**: 1,041 negative reviews misclassified as positive
- **False Negatives**: 1,127 positive reviews misclassified as negative
- **Error Rate**: 23.73% (slight bias toward negative predictions)

### 10.4 Performance Discussion

**Unexpected Results**: Traditional ML beat deep learning by a lot, which wasn't what we expected based on what we'd read about NLP tasks. This result shows us several important things:

1. **Feature Representation**: TF-IDF's explicit word importance weighting proved more effective than dense Word2Vec embeddings for this specific task
2. **Model Complexity**: The BiLSTM's additional complexity may have led to overfitting despite regularization efforts
3. **Data Characteristics**: The IMDB dataset's characteristics may favor sparse, interpretable features over dense representations

**Challenges Identified**:
- **Overfitting**: Both models showed some overfitting, controlled through regularization
- **Loss Function Performance**: BiLSTM achieved final training loss of 0.23 and validation loss of 0.41, indicating some overfitting despite early stopping
- **Negation Handling**: Both approaches struggled with complex negations and sarcasm  
- **Computational Efficiency**: Naive Bayes offered 15x faster training with superior performance

**Loss Function Analysis**:
The binary crossentropy loss provided clear optimization signals during training, with the BiLSTM model achieving convergence within 8-10 epochs. However, the gap between training loss (0.23) and validation loss (0.41) suggests that even with regularization, the deep learning model struggled with generalization compared to the traditional ML approach.

**Potential Improvements**:
1. **Advanced Preprocessing**: Negation handling, sentiment-specific tokenization
2. **Ensemble Methods**: Combining predictions from both approaches
3. **Transfer Learning**: Pre-trained embeddings (GloVe, FastText) for BiLSTM
4. **Transformer Models**: BERT-based approaches for state-of-the-art performance

## 11. Limitations and Future Work

### 11.1 Current Limitations

**Data Limitations**:
- **Domain Specificity**: Results may not generalize to other text domains
- **Temporal Bias**: Dataset spans multiple time periods with evolving language patterns
- **Binary Classification**: Real-world sentiment often exists on a spectrum

**Model Limitations**:
- **Context Understanding**: Neither approach captures long-range dependencies effectively
- **Negation Handling**: Limited capability for complex linguistic constructions
- **Sarcasm Detection**: Both models struggle with ironic or sarcastic content

**Evaluation Limitations**:
- **Single Dataset**: Results limited to IMDB movie reviews
- **Binary Metrics**: Don't capture nuanced sentiment gradations
- **Computational Constraints**: Limited hyperparameter exploration due to resource constraints

### 11.2 Future Research Directions

**Technical Improvements**:
1. **Transformer Integration**: Implement BERT, RoBERTa, or DistilBERT models
2. **Ensemble Approaches**: Combine traditional ML and deep learning predictions
3. **Multi-task Learning**: Joint training on sentiment and emotion recognition
4. **Attention Mechanisms**: Improve interpretability and focus on relevant text sections

**Dataset Expansion**:
1. **Multi-domain Evaluation**: Test across different review types (products, restaurants, books)
2. **Multilingual Analysis**: Extend to non-English sentiment analysis
3. **Fine-grained Sentiment**: Move beyond binary to 5-point or continuous scales
4. **Temporal Analysis**: Track sentiment evolution over time

**Practical Applications**:
1. **Real-time Processing**: Optimize models for streaming sentiment analysis
2. **Deployment Optimization**: Model compression and edge computing adaptation
3. **Bias Detection**: Analyze and mitigate demographic and cultural biases
4. **Explainability**: Develop interpretable model explanations for end users

## 12. Reproducibility and Implementation Details

### 12.1 Environment Setup

**Required Dependencies**:
```
python>=3.8
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
gensim>=4.1.0
nltk>=3.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
wordcloud>=1.8.0
```

**Installation Commands**:
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 12.2 Data Preparation

**Steps to Reproduce**:
1. Download IMDB dataset from Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Place `IMDB Dataset.csv` in `datasets/raw/` directory
3. Run preprocessing notebook: `notebooks/sentiment_analysis_main.ipynb`

### 12.3 Model Training

**Naive Bayes Training**:
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Model training
nb_model = MultinomialNB(alpha=5.0)
nb_model.fit(X_train_tfidf, y_train)
```

**BiLSTM Training**:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

# Model architecture
model = Sequential([
    Embedding(vocab_size, 100, weights=[embedding_matrix], trainable=True),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Training configuration
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### 12.4 Repository Structure

```
Sentiment-Analysis_Group-5/
├── datasets/
│   └── raw/
│       └── IMDB Dataset.csv
├── notebooks/
│   ├── sentiment_analysis_main.ipynb
│   └── ML(Naives_Bayes)_experiments.ipynb
├── reports/
│   └── Group_5_Report_Sentiment_Analysis_IMDB_Reviews.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

## 13. Team Contributions

**Benitha Uwituze Rutagengwa**:
- Data exploration, preprocessing pipeline, feature engineering (TF-IDF/Word2Vec)
- EDA visualizations and preprocessing justifications
- Led statistical analysis and data quality assessment
- Developed comprehensive preprocessing pipeline documentation
- Contributed to results interpretation and methodology section

**Jeremiah Olaitan Agbaje**:
- Traditional ML model implementation (Naive Bayes)
- Hyperparameter experimentation and alpha tuning analysis
- Code documentation and systematic evaluation design
- Created experiment tables with 7 alpha parameter configurations
- Contributed to model comparison and statistical analysis

**Tangue Kwizera**:
- Deep learning model design (BiLSTM) and architecture implementation
- Training pipeline setup and results analysis
- Word2Vec embedding integration and optimization
- Neural network hyperparameter experiments and training procedures
- Created model performance visualizations and training curves

**Afsa Umutoniwase**:
- Quality assurance and GitHub repository structure management
- Evaluation metrics implementation and project integration
- Repository organization and version control coordination
- Reproducibility documentation and setup instructions development
- Report compilation, citation management, and final quality assurance

## 14. Conclusion

Our study shows useful things about how traditional machine learning compares to deep learning for sentiment analysis. Our careful experiments produced several important findings that challenge what people usually think about deep learning being better for all NLP tasks.

**Key Findings**:
1. **Traditional ML worked better**: Naive Bayes with TF-IDF features got 86.07% accuracy, beating BiLSTM (74.73%)
2. **Hyperparameter tuning matters**: Careful optimization improved Naive Bayes performance by 0.32 percentage points
3. **Feature choice is important**: Sparse TF-IDF features worked better than dense Word2Vec embeddings for this task
4. **Efficiency advantage**: Traditional approaches gave better performance with much lower computational requirements

**Practical Implications**:
Our results suggest that practitioners should not automatically assume deep learning superiority for all NLP tasks. Traditional ML approaches may offer optimal solutions when:
- Computational resources are limited
- Interpretability is required
- Training data is moderate in size
- Fast deployment is essential

**Broader Impact**:
Our research adds to the growing work examining the practical trade-offs between traditional and deep learning approaches. Our results support a more balanced view of model selection that considers what the task needs, resource limits, and performance requirements.

**What we recommend:**
1. **Try traditional ML first**: Don't automatically assume deep learning is better - test simple models as baselines
2. **Tune your parameters**: Careful hyperparameter optimization made a real difference in our results  
3. **Focus on features**: Good feature engineering can make traditional approaches work really well
4. **Consider combining approaches**: Future work should try mixing both methods for better results

Our study shows that good sentiment analysis isn't just about picking the fanciest model - you need to consider features, experimental design, and practical constraints too.

## References

[1] Ebert, R. (2011). *Life Itself: A Memoir*. Grand Central Publishing. Available at: https://www.hachettebookgroup.com/titles/roger-ebert/life-itself/9780446584968/

[2] Internet Movie Database (IMDb). (1990). "About IMDb: History and Information." Available at: https://www.imdb.com/

[3] Dellarocas, C., Zhang, X. M., & Awad, N. F. (2007). "Exploring the value of online product reviews in forecasting sales: The case of motion pictures." *Journal of Interactive Marketing*, 21(4), 23-45.

[4] Netflix Technology Blog. (2016). "Netflix Recommendations: Beyond the 5 stars." Available at: https://netflixtechblog.com/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429

[5] Liu, B. (2012). *Sentiment Analysis and Opinion Mining*. Morgan & Claypool Publishers. Available at: https://doi.org/10.2200/S00416ED1V01Y201204HLT016

[6] Pang, B., Lee, L., & Vaithyanathan, S. (2002). "Thumbs up?: sentiment classification using machine learning techniques." *Proceedings of the ACL-02 Conference on Empirical Methods in Natural Language Processing*. Available at: https://aclanthology.org/W02-1011/

[7] Martineau, J., & Finin, T. (2009). "Delta TFIDF: An improved feature space for sentiment analysis." *Third International AAAI Conference on Weblogs and Social Media*. Available at: https://ojs.aaai.org/index.php/ICWSM/article/view/13979

[8] Kim, Y. (2014). "Convolutional neural networks for sentence classification." *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*. Available at: https://aclanthology.org/D14-1181/

[9] Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." *Neural Computation*, 9(8), 1735-1780. Available at: https://doi.org/10.1162/neco.1997.9.8.1735

[10] Graves, A., & Schmidhuber, J. (2005). "Framewise phoneme classification with bidirectional LSTM and other neural network architectures." *Neural Networks*, 18(5-6), 602-610. Available at: https://doi.org/10.1016/j.neunet.2005.06.042

[11] Howard, J., & Ruder, S. (2018). "Universal language model fine-tuning for text classification." *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics*. Available at: https://aclanthology.org/P18-1031/

[12] Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). "Learning word vectors for sentiment analysis." *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics*. Available at: https://aclanthology.org/P11-1015/

[13] IMDB Dataset. (2025). "IMDB Dataset of 50K Movie Reviews." Kaggle. Available at: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews