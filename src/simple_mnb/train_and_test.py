import glob
import math
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn import metrics

def train_and_test(trainPath, validationPath=None, testPath=None, tfidf=False, stem=False, laplace=0.001, outputFile=None):
    stop_words = set(stopwords.words('english'))
    ss = SnowballStemmer(language='english')
    totalDocs=0 #total number of docs
    vocabSize = 0 #total number of unique words - will be same size as keys of vocab{}
    vocab = {} #holds all words and their frequencies
    wordPerCat = {}; docPerCat = {}; vocabSizePerCat = {}; numDocsWithTerm = {}
    for folder in sorted(glob.glob(trainPath)):
        folderList = folder.split('_')
        classLetter = folderList[2]
        dicForFolder = {}
        for filename in sorted(glob.glob(folder+"/*.res")):
            totalDocs+=1 #incremement total num docs
            try:
                docPerCat[classLetter]+=1 #increment docs per category
            except:
                docPerCat[classLetter]=1
            totalNumWordsInDoc = 0
            f=open(filename, errors='ignore', encoding='cp1252')
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
        for i in wordPerCat: #calculates tf-idf for each word per class in wordPerCat
            for key, value in wordPerCat[i].items():
                deted = int(numDocsWithTerm[key])
                wordPerCat[i][key] = float(float(wordPerCat[i][key]) * (math.log(totalDocs/deted)))

    #all prior probs for each class
    priorsDict = {}
    for key in docPerCat.keys():
        priorsDict[key] = math.log(float(docPerCat[key]/totalDocs)) #prior probability of any given class

    # VALIDATION
    if (validationPath):
        count, numRight, numWrong = 0, 0, 0
        y_true, y_pred = [], []
        arrayforvalidation = []

        for folder in sorted(glob.glob(validationPath)):
            folderList = folder.split('_')
            classLetter = folderList[2] #will be A, B, C, . . . or I
            for filename in sorted(glob.glob(folder+"/*.res")):
                count+=1
                allPWC = {}
                f=open(filename, errors='ignore', encoding='cp1252')
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
                            for key in docPerCat.keys():
                                if line[0] in wordPerCat[key]:
                                    c_in_c = float(wordPerCat[key][line[0]]) #count in class
                                else:
                                    c_in_c = 0.0
                                try: #get probablity of class given word
                                    allPWC[key].append(math.log(float((c_in_c + laplace)/((vocabSize) + vocabSizePerCat[key]))))
                                except:
                                    allPWC[key] = [math.log(float((c_in_c + laplace)/((vocabSize) + vocabSizePerCat[key])))]
                            i+=1
            
                dictOfClassProb = {}
                for key in docPerCat.keys():
                    dictOfClassProb[key] = priorsDict[key] + sum(allPWC[key])
                    
                classPredicted = max(dictOfClassProb, key=dictOfClassProb.get)
                y_true.append(classLetter)
                y_pred.append(classPredicted)
                if classPredicted == classLetter:
                    numRight+=1
                    arrayforvalidation.append("CORRECT /// Class: " + classLetter + "; Predicted: " + classPredicted + "; Total accuracy: " + str((numRight/count)*100))
                else:
                    arrayforvalidation.append("WRONG /// Class: " + classLetter + "; Predicted: " + classPredicted + "; Total accuracy: " + str((numRight/count)*100))
                
        print('CONFUSION MATRIX \n\n' + str(metrics.confusion_matrix(y_true, y_pred)))
        print('\n\nCLASSIFICATION REPORT \n\n' + str(metrics.classification_report(y_true, y_pred, digits=3)))
        with open("outputvalidation.txt", 'w') as file:
            for i in range(len(arrayforvalidation)):
                file.write(arrayforvalidation[i] + '\n')

    #TEST
    if (testPath):
        count = 0; catKeyCount = 0; arraytowrite = []
        filenamearray = []

        files = sorted(glob.glob(testPath + '/*.res'), key=len)
        for filename in files:
            f=open(filename, errors='ignore', encoding='cp1252')
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
                        for key in docPerCat.keys():
                            if line[0] in wordPerCat[key]:
                                c_in_c = float(wordPerCat[key][line[0]]) #count in class
                            else:
                                c_in_c = 0.0
                            try:
                                allPWC[key].append(math.log(float((c_in_c + laplace)/((vocabSize) + vocabSizePerCat[key]))))
                            except:
                                allPWC[key] = [math.log(float((c_in_c + laplace)/((vocabSize) + vocabSizePerCat[key])))]
                        i+=1
                            
            dictOfClassProb = {}
            for key in docPerCat.keys():
                    dictOfClassProb[key] = priorsDict[key] + sum(allPWC[key])
                    
            classPredicted = max(dictOfClassProb, key=dictOfClassProb.get)
            print("Filename: " + str(filename) + " // Predicted Class: " + classPredicted)
            arraytowrite.append(classPredicted)

        #write to file the predictions for the test data
        if (outputFile):
            with open(outputFile, 'w') as file:
                for k in range(len(arraytowrite)):
                    file.writelines("Filename: " + str(filenamearray[k]) + ' // Predicted Class: ' + str(arraytowrite[k]) + '\n')
        
        return arraytowrite

train_and_test('src/simple_mnb/data/train/class_*_train', validationPath='src/simple_mnb/data/validation/class_*_validation', testPath='src/simple_mnb/data/test/test', stem=True, outputFile='output.txt')