import glob
import math
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn import metrics

# Utils
from src.util.utils import *

'''

Multinomial Naive Bayes - One method for training and testing, optional parameters - one line of code to train and test the classifier

Usage example:
import simple_mnb.simple_mnb as mnb
mnb.train_and_test('./training/files/', './validation/files/')

Specify paths for training and testing data as folders containing subfolders, each with the name of their class label
Example: for class labels A, B, and C, specify './data/' if /data contains subfolders each called /A, /B, and /C, with
each folder containing individual documents (.txt, .res)

document should be in BOW format (i.e., each line is a given word and its number of occurences, followed by a newline character)

currently no support for non-BOW format // TO-DO - more file type and format handling

Optional paramters:
    set fileType to be txt or res, default is res
    set tfidf=True to use TF-IDF vectorization instead of count vectorization (default)
    set stem=True to use SnowballStemmer on tokens
    try different laplace value to tune classifier (0.1, 0.05, 0.01, 0.005, 0.001, etc.)
    set outputFile=True to write predicted labels for some test or validation data (per document) to a single text file

'''

def train_and_test(trainPath, validationPath=None, testPath=None, fileType='res', tfidf=False, stem=False, laplace=0.001, outputFile=None):
    # Stop words and stemmer
    stop_words = set(stopwords.words('english'))
    ss = SnowballStemmer(language='english')
    totalDocs=0 #total number of docs
    vocabSize = 0 #total number of unique words - will be same size as keys of vocab{}
    vocab = {} #holds all words and their frequencies
    wordPerCat = {}; docPerCat = {}; vocabSizePerCat = {}; numDocsWithTerm = {}
    if (fileType == 'res'):
        encoding = 'cp1252'
    elif (fileType == 'txt'):
        encoding = 'utf-8'
    else:
        print("Error. Unsupported file type.")
    for folder in sorted(glob.glob(trainPath + '*')):
        classLetter = folder[-1]
        dicForFolder = {}
        for filename in sorted(glob.glob(folder+"/*." + fileType)):
            totalDocs+=1 #incremement total num docs
            try:
                docPerCat[classLetter]+=1 #increment docs per category
            except:
                docPerCat[classLetter]=1
            totalNumWordsInDoc = 0
            f=open(filename, errors='ignore', encoding=encoding)
            for line in f:
                line = line.rstrip()
                line = line.split(" ")
                line[0] = re.sub("[^a-zA-Z]+", "", line[0])
                line[0] = line[0].lower()
                if not (line[0] == "" or len(line) != 2 or line[0] in stop_words):
                    if line[0] in vocab: #word exists in vocabulary
                        vocab[line[0]] += int(line[1])
                        totalNumWordsInDoc += int(line[1])
                        numDocsWithTerm[line[0]]+=1
                        if line[0] in dicForFolder:
                            dicForFolder[line[0]] += int(line[1])
                        else:
                            dicForFolder[line[0]] = int(line[1])
                            try:
                                vocabSizePerCat[classLetter]+=1
                            except:
                                vocabSizePerCat[classLetter]=1
                    else: #new word
                        #add to vocab, increment vocabsize
                        if (stem):
                            line[0] = ss.stem(line[0])
                        vocabSize+=1
                        vocab[line[0]] = int(line[1])
                        dicForFolder[line[0]] = int(line[1])
                        try:
                            vocabSizePerCat[classLetter]+=1
                        except:
                            vocabSizePerCat[classLetter]=1
                        totalNumWordsInDoc += int(line[1])
                        numDocsWithTerm[line[0]] = 1
        wordPerCat[classLetter] = dicForFolder

    if (tfidf):
        wordPerCat = tfidf_calc(wordPerCat, numDocsWithTerm, totalDocs)
    
    #all prior probs for each class
    priorsDict = prior_probs_calc(docPerCat, totalDocs)

    # VALIDATION
    if (validationPath):
        count, numRight, numWrong = 0, 0, 0
        y_true, y_pred = [], []
        arrayforvalidation = []

        for folder in sorted(glob.glob(validationPath + '*')):
            classLetter = folder[-1]
            for filename in sorted(glob.glob(folder+"/*." + fileType)):
                count+=1
                allPWC = {}
                f=open(filename, errors='ignore', encoding=encoding)
                for line in f:
                    line = line.rstrip()
                    line = line.split(" ")
                    line[0] = re.sub("[^a-zA-Z]+", "", line[0])
                    line[0] = line[0].lower()
                    if not (line[0] == "" or len(line) != 2 or line[0] in stop_words):
                        i=1
                        if (stem):
                            line[0] = ss.stem(line[0])
                        while i <= int(line[1]):
                            allPWC = pwc_calc(allPWC, line[0], docPerCat, wordPerCat, laplace, vocabSize, vocabSizePerCat)
                            i+=1
            
                dictOfClassProb = class_prob_calc(allPWC, priorsDict)
                    
                y_true, y_pred, arrayforvalidation, numRight = predict_class(dictOfClassProb, y_true, classLetter, y_pred, numRight, arrayforvalidation, count)
                
        print('CONFUSION MATRIX \n\n' + str(metrics.confusion_matrix(y_true, y_pred)))
        print('\n\nCLASSIFICATION REPORT \n\n' + str(metrics.classification_report(y_true, y_pred, digits=3)))
        if (outputFile):
            with open("outputvalidation.txt", 'w') as file:
                for i in range(len(arrayforvalidation)):
                    file.write(arrayforvalidation[i] + '\n')

    #TEST
    if (testPath):
        count = 0; catKeyCount = 0; arraytowrite = []
        filenamearray = []

        files = sorted(glob.glob(testPath + '/*.' + fileType), key=len)
        for filename in files:
            f=open(filename, errors='ignore', encoding=encoding)
            filenamearray.append(str(filename))
            allPWC = {}
            for line in f:
                line = line.rstrip()
                line = line.split(" ")
                line[0] = re.sub("[^a-zA-Z]+", "", line[0])
                line[0] = line[0].lower()
                if not(line[0] == "" or len(line) != 2 or line[0] in stop_words):
                    i=1
                    if (stem):
                        line[0] = ss.stem(line[0])
                    while i <= int(line[1]):
                        allPWC = pwc_calc(allPWC, line[0], docPerCat, wordPerCat, laplace, vocabSize, vocabSizePerCat)
                        i+=1
                            
            dictOfClassProb = class_prob_calc(allPWC, priorsDict)
                    
            classPredicted = max(dictOfClassProb, key=dictOfClassProb.get)
            print("Filename: " + str(filename) + " // Predicted Class: " + classPredicted)
            arraytowrite.append(classPredicted)

        #write to file the predictions for the test data
        if (outputFile):
            with open(outputFile, 'w') as file:
                for k in range(len(arraytowrite)):
                    file.writelines("Filename: " + str(filenamearray[k]) + ' // Predicted Class: ' + str(arraytowrite[k]) + '\n')
        
        return arraytowrite

    if not (testPath or validationPath):
        print("Please specify a directory of validation data or test data in order to test this classifier.")

