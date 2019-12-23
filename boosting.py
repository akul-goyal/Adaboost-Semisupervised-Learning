import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import random

import warnings
import math

def adaboost_clf(Y_train, X_train, M, err_pos, err_neg, noiseLabel):
    clf = DecisionTreeClassifier(max_depth=1)
    n_train = len(X_train)
    n_classes = len(X_train[0])
    # Initialize weights
    w = np.ones(n_train)
    w = w.tolist()
    clf_list = []

    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = clf.predict(X_train)

        miss = []
        count = 0
        # Indicator function
        for x in range(len(pred_train_i)):
            if pred_train_i[x]==Y_train[x]:
                miss.append(0)
            else:
                miss.append(1)

        # Error
        err_m = 0
        for x in range(len(miss)):
            err_m += miss[x]*w[x]

        err_m /= sum(w)

        if err_m ==0:
            err_m = .0001
        # Divisor
        dev = 1 - err_pos - err_neg

        # Alpha
        alpha_m = .01 * np.log((1 - err_m) / float(err_m))

        #noisy correction
        miss2 = []
        for x in range(len(miss)):
            if noiseLabel[x]==1:
                if miss[x] == 0:
                   if Y_train[x] > 0:
                      miss2.append((1 - err_neg + err_pos)/dev)
                   else:
                       miss2.append((1 + err_neg - err_pos)/dev)
                else:
                    if Y_train[x] > 0:
                        miss2.append((-(1 - err_neg + err_pos))/dev)
                    else:
                        miss2.append((-(1 + err_neg - err_pos))/dev)
            else:
                if miss[x]==0:
                    miss2.append(1)
                else:
                    miss.append(-1)


        for x in range(len(miss2)):
            temp = miss2[x]
            temp *= (-alpha_m)
            temp = math.exp(temp)
            temp *= w[x]
            w[x] = temp
        #w = np.multiply(w, np.exp([(float(x)/dev) * (-alpha_m) for x in miss2]))
        clf_list.append((clf, alpha_m))
    return clf_list

def classify_adaBoosting(clf_list, X):
    total = 0
    for clf in clf_list:
        if clf[0].predict(X)>0:
            total += clf[1]
        else:
            total -= clf[1]
    if total > 0:
        return 1
    else:
        return 0
def calcError (confusion_matrixs, ones, agree, percentone, percentzero):
    for i in range(len(confusion_matrixs)):
        #confusion matrix        0  1
        zero = confusion_matrixs[i][0] #0
        one =  confusion_matrixs[i][1] #1

        false_negative_confusion = one[0]
        false_postive_confusion = zero[1]
        if false_negative_confusion != 0 and false_postive_confusion != 0:
            break
    e0 = ((-1*(ones)*(1-false_postive_confusion)) + agree)/((-0.5)*(1-false_postive_confusion-false_negative_confusion))
    e1 = (((false_negative_confusion*ones)-agree)/(0.5*(1-false_postive_confusion-false_negative_confusion)))+1
    return e0,e1

def probab(answers, noisy_labels, clfNum):
    #getting prob of x classifier
    yw = []
    agree = 0
    ones = 0
    total=0
    for ptn in range(len(answers)):
        yw.append(answers[ptn][clfNum][0])
        total +=1
    for num in range(len(yw)):
        if noisy_labels[num] == 1:
            ones += 1
            if yw[num] == noisy_labels[num]:
                agree += 1
    return ones/total, agree/total
    
