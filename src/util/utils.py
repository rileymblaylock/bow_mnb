import math

def tfidf_calc(wordPerCat, numDocsWithTerm, totalDocs):
    for i in wordPerCat:
        for key, value in wordPerCat[i].items():
            deted = int(numDocsWithTerm[key])
            wordPerCat[i][key] = float(float(wordPerCat[i][key]) * (math.log(totalDocs/deted)))
    return wordPerCat

def prior_probs_calc(docPerCat, totalDocs):
    priorsDict = {}
    for key in docPerCat.keys():
        priorsDict[key] = math.log(float(docPerCat[key]/totalDocs))
    return priorsDict

def pwc_calc(allPWC, word, docPerCat, wordPerCat, laplace, vocabSize, vocabSizePerCat):
    for key in docPerCat.keys():
        if word in wordPerCat[key]:
            c_in_c = float(wordPerCat[key][word]) #count in class
        else:
            c_in_c = 0.0
        try: #get probablity of class given word
            allPWC[key].append(math.log(float((c_in_c + laplace)/((vocabSize) + vocabSizePerCat[key]))))
        except:
            allPWC[key] = [math.log(float((c_in_c + laplace)/((vocabSize) + vocabSizePerCat[key])))]
    return allPWC

def class_prob_calc(allPWC, priorsDict):
    dictOfClassProb = {}
    for key in allPWC.keys():
        dictOfClassProb[key] = priorsDict[key] + sum(allPWC[key])
    return dictOfClassProb

def predict_class(dictOfClassProb, y_true, classLetter, y_pred, numRight, arrayforvalidation, count):
    classPredicted = max(dictOfClassProb, key=dictOfClassProb.get)
    y_true.append(classLetter)
    y_pred.append(classPredicted)
    if classPredicted == classLetter:
        numRight+=1
        arrayforvalidation.append("CORRECT /// Class: " + classLetter + "; Predicted: " + classPredicted + "; Total accuracy: " + str((numRight/count)*100))
    else:
        arrayforvalidation.append("WRONG /// Class: " + classLetter + "; Predicted: " + classPredicted + "; Total accuracy: " + str((numRight/count)*100))
    return y_true, y_pred, arrayforvalidation, numRight