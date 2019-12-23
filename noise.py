
import pandas as pd
import numpy as np

from sklearn import preprocessing

import random

import boosting, read, errorTest

def fxn():
    test, unlabel, label, true, unlabel_unmod = read.read(file='data.csv', drop=['id'], retNum = 0)

    #add noise to the unlabelled data
    noise = {}
    false_pos = 0
    false_neg = 0
    for row in unlabel_unmod:
        #first pick whether it should be wrong or right:
        if (random.randint(1, 11))>6: #we are keeping it at 60% correct 40% noise
            #false negative
            if row[0]>0:
                row[0] = 0
                false_neg += 1
            #false postive
            else:
                row[0] = 1
                false_pos += 1
    
    #create mapping between whether noisy or not
    mapping = {}
    for row in label:
        mapping[tuple(row)] = 0
    for row in unlabel_unmod:
        mapping[tuple(row)] = 1
    
    noiseLabel = []
    Y_train = []
    X_train = []
    #combine the noisy and clean dataset
    tempList = label + unlabel_unmod
    
    #mix them together
    random.shuffle(tempList)

    #bring their x,y,noise label to be sent to boosting
    for row in tempList:
        noiseLabel.append(mapping[tuple(row)])
        X_train.append(row[1:])
        Y_train.append(row[:1][0])
    clf_list = boosting.adaboost_clf(Y_train, X_train, 20, false_pos/len(unlabel_unmod), false_neg/len(unlabel_unmod), noiseLabel)
    return errorTest.test(clf_list, test)
ret = 0
avg = 0
for x in range(50):
    ret = fxn()
    if ret>.50:
        ret = 1-ret
    avg += ret
avg = avg/50
print(avg)

