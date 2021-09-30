# bow_mnb
Given training data in BOW format, trains a simple multinomial Naive Bayes classifier

Multinomial Naive Bayes - One method for training and testing, optional parameters - one line of code to train and test the classifier

Usage example:
First download and install the package in your terminal:
python3 -m pip install bow_mnb

Then in your python file:
import bow_mnb.mnb as mnb
mnb.train_and_test('./training/files/', './validation/files/')

Specify paths for training and testing data as folders containing subfolders, each with the name of their class label
Example: for class labels A, B, and C, specify './trainingdata/' if /trainingdata contains subfolders each called /A, /B, and /C, with
each folder containing individual documents (.txt, .res), and the same format for testing and/or validation data

document should be in BOW format (i.e., each line is a given word and its number of occurences, followed by a newline character)

currently no support for non-BOW format // TO-DO - more file type and format handling

Optional paramters:
    set fileType to be txt or res, default is txt
    set tfidf=True to use TF-IDF vectorization instead of count vectorization (default)
    set stem=True to use SnowballStemmer on tokens
    try different laplace value to tune classifier (0.1, 0.05, 0.01, 0.005, 0.001, etc.)
    set outputFile=True to write predicted labels for some test or validation data (per document) to a single text file
