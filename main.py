#Riley Blaylock

import glob
import math
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import os
from os import walk
os.chdir('C:/Users/riley/Desktop/ML/mlproj1/')

stop_words = set(stopwords.words('english'))
ss = SnowballStemmer(language='english')
totalDocs=0 #total number of docs
docPerCat = { #total counts of docs per catergory
    'A': 0,'B': 0,'C': 0,'D': 0,'E': 0,'F': 0,'G': 0,'H': 0, 'I': 0 }
catKey = { #this is used so the for loop can know what cat its in
    1: 'A',2: 'B',3: 'C',4: 'D',5: 'E',6: 'F',7: 'G',8: 'H',9: 'I' }
vocabSize = 0 #total number of unique words - will be same size as keys of vocab{}
vocab = {} #holds all words and their frequencies
wordPerCat = { #holds all words and their frequencies per category
    'A':{},'B':{},'C':{},'D':{},'E':{},'F':{},'G':{},'H':{},'I':{} }
vocabSizePerCat = docPerCat
numDocsWithTerm = {}
catKeyCount = 0 #temp variable to use as key-value finder for intra-loop class labeling
path = 'C:/Users/riley/Desktop/ML/mlproj1/'

for folder in sorted(glob.glob(path+'Train/class_*_train')):
    catKeyCount+=1
    classLetter = catKey[catKeyCount] #will be A, B, C, . . . or I
    dicForFolder = {}
    #incremement key for value catKeyCount = docPerCat
    for filename in sorted(glob.glob(folder+"/*.res")):
        totalDocs+=1 #incremement total num docs
        docPerCat[classLetter]+=1 #increment docs per category
        totalNumWordsInDoc = 0
        f=open(filename, errors='ignore', encoding='cp1252')
        for line in f:
            line = line.rstrip()
            line = line.split(" ")
            line[0] = re.sub("[^a-zA-Z]+", "", line[0])
            line[0] = line[0].lower()
            if line[0] == "" or len(line) != 2 or line[0] in stop_words: #bad line
                tmp = 0 #do nothing
            elif line[0] in vocab: #word exists in vocabulary
                #add to wordPerCat and increment key based on line[1]
                tempNum = vocab[line[0]]
                tempNum += int(line[1])
                vocab[line[0]] = tempNum
                totalNumWordsInDoc += int(line[1])
                numDocsWithTerm[line[0]]+=1
                if line[0] in dicForFolder:
                    tempNum3 = dicForFolder[line[0]]
                    tempNum3 += int(line[1])
                    dicForFolder[line[0]] = tempNum3
                else:
                    dicForFolder[line[0]] = int(line[1])
                    vocabSizePerCat[classLetter]+=1
            else: #new word
                #add to vocab, increment vocabsize
                line[0] = ss.stem(line[0])
                vocabSize+=1
                vocab[line[0]] = int(line[1])
                dicForFolder[line[0]] = int(line[1])
                #add to wordPerCat and increment key based on line[1]
                vocabSizePerCat[classLetter]+=1
                totalNumWordsInDoc += int(line[1])
                numDocsWithTerm[line[0]] = 1
        #20000 = 84.1
        if docPerCat[classLetter] > 20000:
            break
        else:
            tmp = 0
    wordPerCat[classLetter] = dicForFolder

for i in docPerCat:
    print(docPerCat[i])

for i in wordPerCat: #calculates tf-idf for each word per class in wordPerCat
    for key, value in wordPerCat[i].items():
        deted = int(numDocsWithTerm[key])
        wordPerCat[i][key] = float(float(wordPerCat[i][key]) * (math.log(totalDocs/deted)))

def get_P_of_c(classLetter): #prior probability of any given class
    return float(docPerCat[classLetter]/totalDocs)

def count_in_class(classLetter, word): #the count of a word in its class
    if word in wordPerCat[classLetter]:
        return float(wordPerCat[classLetter][word])
    else:
        return 0.0

def get_p_of_w_given_c(classLetter, word): #get probablity of class given word
    return float(((count_in_class(classLetter, word)) + 0.01)/((vocabSize) + vocabSizePerCat[classLetter])) #82.3% with 0.01 laplace value

#all prior probs for each class
priorA = math.log(get_P_of_c('A')); priorB = math.log(get_P_of_c('B')); priorC = math.log(get_P_of_c('C'))
priorD = math.log(get_P_of_c('D')); priorE = math.log(get_P_of_c('E')); priorF = math.log(get_P_of_c('F'))
priorG = math.log(get_P_of_c('G')); priorH = math.log(get_P_of_c('H')); priorI = math.log(get_P_of_c('I'))

count, numRight, catKeyCount = 0, 0, 0
arrayforvalidation = []

for folder in sorted(glob.glob(path+'Validation/class_*_validation')):
    catKeyCount+=1
    classLetter = catKey[catKeyCount] #will be A, B, C, . . . or I
    for filename in sorted(glob.glob(folder+"/*.res")):
        count+=1
        Apwc, Bpwc, Cpwc, Dpwc, Epwc, Fpwc, Gpwc, Hpwc, Ipwc = [], [], [], [], [], [], [], [], []
        f=open(filename, errors='ignore', encoding='cp1252')
        for line in f:
            #handle errors first
            line = line.rstrip()
            line = line.split(" ")
            line[0] = re.sub("[^a-zA-Z]+", "", line[0])
            line[0] = line[0].lower()
            if line[0] == "" or len(line) != 2 or line[0] in stop_words: #bad line
                tmp = 0 #do nothing
            else:
                i=1
                while i <= int(line[1]):
                    line[0] = ss.stem(line[0])
                    Apwc.append(math.log(get_p_of_w_given_c('A', line[0]))); Bpwc.append(math.log(get_p_of_w_given_c('B', line[0]))); Cpwc.append(math.log(get_p_of_w_given_c('C', line[0])))
                    Dpwc.append(math.log(get_p_of_w_given_c('D', line[0]))); Epwc.append(math.log(get_p_of_w_given_c('E', line[0]))); Fpwc.append(math.log(get_p_of_w_given_c('F', line[0])))
                    Gpwc.append(math.log(get_p_of_w_given_c('G', line[0]))); Hpwc.append(math.log(get_p_of_w_given_c('H', line[0]))); Ipwc.append(math.log(get_p_of_w_given_c('I', line[0])))
                    i+=1
       
        probforClassA = priorA + sum(Apwc); probforClassB = priorB + sum(Bpwc); probforClassC = priorC + sum(Cpwc)
        probforClassD = priorD + sum(Dpwc); probforClassE = priorE + sum(Epwc); probforClassF = priorF + sum(Fpwc)
        probforClassG = priorG + sum(Gpwc); probforClassH = priorH + sum(Hpwc); probforClassI = priorI + sum(Ipwc)

        probforClassList = [probforClassA, probforClassB, probforClassC, probforClassD, 
                probforClassE, probforClassF, probforClassG, probforClassH, probforClassI]
        
        dictOfClassProb = {
            probforClassA: 'A',probforClassB: 'B',probforClassC: 'C',probforClassD: 'D',
            probforClassE: 'E',probforClassF: 'F',probforClassG: 'G',probforClassH: 'H',probforClassI: 'I' }
      
        classPredicted = dictOfClassProb[max(probforClassList)]
        if str(classPredicted) == classLetter:
            numRight+=1
            print("CORRECT /// Class: " + str(classLetter) + "; Predicted: " + str(classPredicted) + "; Total accuracy: " + str((numRight/count)*100))
            arrayforvalidation.append(str("CORRECT /// Class: " + str(classLetter) + "; Predicted: " + str(classPredicted) + "; Total accuracy: " + str((numRight/count)*100)))
        else:
            print("WRONG /// Class: " + str(classLetter) + "; Predicted: " + str(classPredicted) + "; Total accuracy: " + str((numRight/count)*100))
            arrayforvalidation.append(str("WRONG /// Class: " + str(classLetter) + "; Predicted: " + str(classPredicted) + "; Total accuracy: " + str((numRight/count)*100)))
        

with open("outputvalidation.txt", 'w') as file:
    for i in range(len(arrayforvalidation)):
        file.write(arrayforvalidation[i] + '\n')

#TEST SET
print("TEST SET")
count = 0; catKeyCount = 0; arraytowrite = []
j = 0
filenamearray = []


files = sorted(glob.glob(path+'/TestSet/Test/*.res'), key=len)
for filename in files:
    Apwc, Bpwc, Cpwc, Dpwc, Epwc, Fpwc, Gpwc, Hpwc, Ipwc = [], [], [], [], [], [], [], [], []
    #with open(os.path.join('C:/Users/riley/Desktop/ML/mlproj1/TestSet/Test/', filename), errors='ignore', encoding='cp1252') as f:
    f=open(filename, errors='ignore', encoding='cp1252')
    filenamearray.append(str(filename))
    for line in f:
        #handle errors first
        line = line.rstrip()
        line = line.split(" ")
        line[0] = re.sub("[^a-zA-Z]+", "", line[0])
        line[0] = line[0].lower()
        if line[0] == "" or len(line) != 2 or line[0] in stop_words: #bad line
            tmp = 0 #do nothing  
        else:
            i=1
            while i <= int(line[1]):
                line[0] = ss.stem(line[0])
                Apwc.append(math.log(get_p_of_w_given_c('A', line[0]))); Bpwc.append(math.log(get_p_of_w_given_c('B', line[0]))); Cpwc.append(math.log(get_p_of_w_given_c('C', line[0])))
                Dpwc.append(math.log(get_p_of_w_given_c('D', line[0]))); Epwc.append(math.log(get_p_of_w_given_c('E', line[0]))); Fpwc.append(math.log(get_p_of_w_given_c('F', line[0])))
                Gpwc.append(math.log(get_p_of_w_given_c('G', line[0]))); Hpwc.append(math.log(get_p_of_w_given_c('H', line[0]))); Ipwc.append(math.log(get_p_of_w_given_c('I', line[0])))
                i+=1
                    
    probforClassA = priorA + sum(Apwc); probforClassB = priorB + sum(Bpwc); probforClassC = priorC + sum(Cpwc)
    probforClassD = priorD + sum(Dpwc); probforClassE = priorE + sum(Epwc); probforClassF = priorF + sum(Fpwc)
    probforClassG = priorG + sum(Gpwc); probforClassH = priorH + sum(Hpwc); probforClassI = priorI + sum(Ipwc)

    probforClassList = [probforClassA, probforClassB, probforClassC, probforClassD, 
                probforClassE, probforClassF, probforClassG, probforClassH, probforClassI]
        
    dictOfClassProb = {
            probforClassA: 'A',probforClassB: 'B',probforClassC: 'C',probforClassD: 'D',
            probforClassE: 'E',probforClassF: 'F',probforClassG: 'G',probforClassH: 'H',probforClassI: 'I' }
      
    classPredicted = dictOfClassProb[max(probforClassList)]
    print("Predicted: " + str(classPredicted))
    arraytowrite.append(str(classPredicted))

#write to file the predictions forthe test data
with open("output.txt", 'w') as file:
    for k in range(len(arraytowrite)):
        file.writelines(str(arraytowrite[k]) + '\n')






