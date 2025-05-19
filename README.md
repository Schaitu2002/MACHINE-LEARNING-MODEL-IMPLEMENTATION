# MACHINE-LEARNING-MODEL-IMPLEMENTATION
* COMPANY : CODTECH IT SOLUTIONS
* NAME : SURU CHAITANYA
* INTERN ID : CT08DA121
* DOMAIN : PYTHON PROGRAMMING
* DURATION : 8 WEEKS
* MENTOR : NEELA SANTOSH

## DESCRIPTION OF THIS PROJECT

The goal of this project is to build a machine learning model that can classify SMS messages as spam (unwanted/advertisement messages) or ham (legitimate messages). This can help in filtering out spam messages automatically.

## Dataset Used:
SMS Spam Collection Dataset

Source: UCI Repository / GitHub Link

Format: Tab-separated (.tsv)

## Fields:
label: Indicates if the message is spam or ham

message: The actual content of the SMS

## Model Explanation:
CountVectorizer: Transforms text into a bag-of-words format.

MultinomialNB: Naive Bayes classifier suited for discrete data like word counts.

Train/Test Split: Keeps 80% of the data for training and 20% for testing.

Evaluation: Uses accuracy and a classification report for metrics

## Data Preprocessing Steps:
Load Data: Read the dataset using Pandas.

Label Encoding: Convert labels spam and ham to binary values (1 for spam, 0 for ham).

Text Cleaning: Convert messages to lowercase and remove special characters and punctuation.

Text Vectorization: Use TF-IDF Vectorizer to convert text data into numerical feature vectors.

## Machine Learning Models Used:
We trained and compared the performance of three classification models:

Model	Description
Multinomial Naive Bayes	Good for text classification and word count-based features.
Logistic Regression	Simple linear classifier, efficient for binary outcomes.
Random Forest	Ensemble of decision trees; robust and generally more accurate.

Each model was trained using the TF-IDF features and evaluated using standard metrics.

## Tools & Libraries:
Python

Scikit-learn: Model building and evaluation

Pandas: Data handling

NumPy: Numeric processing

Matplotlib & Seaborn: Visualizations

## Evaluation Metrics:
Accuracy Score

Classification Report (Precision, Recall, F1-score)

Confusion Matrix

Bar Chart: Comparing model performance visually

## OUTPUT
All models showed high accuracy. Logistic Regression and Random Forest generally performed slightly better than Naive Bayes, depending on the random seed and dataset distribution. Visualizations helped to analyze model performance clearly.
