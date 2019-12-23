#!/usr/bin/env python
# coding: utf-8


import boosting, read, classifiers, wrapperDS, errorTest, printOn, shuffle
def preprocess(percentage, basicNN = False):
    printOn.blockPrint()
    if basicNN == True:
        test, unlabel, label, true, x, y, x_true, y_true,x_test,y_test = read.read(file='data.csv', drop=['id'], retNum=1, chopNum=0,
                                                                     unlabel_percentage=percentage, ytrain=True)
    else:
        test, unlabel, label, true, x, y, x_true, y_true = read.read(file='data.csv', drop=['id'], retNum=1, chopNum=0,
                                                                     unlabel_percentage=percentage)
    clfs = classifiers.ensemble(x,y)
    printOn.enablePrint()
    if basicNN == True:
        return unlabel, clfs, true, x,y, test, y_test, x_test
    else:
        return unlabel, clfs, true, x, y, test