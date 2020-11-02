# BagofWordsMultinomialNaiveBayes for text classification
Given a set of training data (emails) in "bag of words" format with corresponding class labels, this model predicts class labels for emails in the test dataset. This model uses multinomial naive Bayes algorithm implemented from scratch, using minimal libraries except nltk for stopwords and SnowballStemmer for preprocessing.
Sub-methods employed:
  - TF-IDF weighting
  - Modified Laplace Smoothing
  - Partial log-sum-exp trick to prevent float underflow
Final accuracy: ~85%
Better feature selection needed for improvement to approach ~90% accuracy
