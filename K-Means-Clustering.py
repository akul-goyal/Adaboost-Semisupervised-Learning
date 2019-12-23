#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import random
import subprocess


def run():
    # read in the data
    df = pd.read_csv('heart.csv')

    # normalize all points between [0,1]
    from sklearn import preprocessing
    x = df.values # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)

    # create 100/100 split between labelled on nonlablled array, 200 unsupervised data set, and 100 validation dataset
    df_array = df.values.tolist()
    df_test = df_array[:103]
    df_array = df_array[103:]
    x = []
    y = []
    xAll = []
    yAll = []

    test_x = []
    test_y = []

    df_nolabels = []
    df_unsupervised = []
    count1 = 0
    count2 = 0
    label_nolabels = {}
    for point in df_array:

        # unlablled 200 points data
        df_unsupervised.append(point[:-1])

        # lablled 200 points data
        xAll.append(point[:-1])
        yAll.append(point[-1:][0])

        # unlablled+ labelled 100 points split data
        if random.randrange(2)==1 and count1<100:
            count1 += 1
            x.append(point[:-1])
            test_x.append(point[:-1])
            y.append(point[-1:][0])
            test_y.append(point[-1:][0])
        elif count2<100:
            count2 += 1
            label_nolabels[tuple(point[:-1])]= point[-1:]
            df_nolabels.append(point[:-1])
        else:
            count1 += 1
            x.append(point[:-1])
            test_x.append(point[:-1])
            y.append(point[-1:][0])
            test_y.append(point[-1:][0])

    # train f1...fn classifiers on labelled data, will use 6 types: decision stumps, knn, svm, guassian mixture mode
    #                                                                native bayes, logistic regression
    clf1 = svm.SVC(kernel='linear', gamma='scale').fit(x, y)
    clf2 = KNeighborsClassifier(n_neighbors=3).fit(x, y)
    clf3 = DecisionTreeClassifier(splitter= 'random').fit(x, y)
    clf4 = DecisionTreeClassifier(splitter= 'random').fit(x, y)
    clf5 = DecisionTreeClassifier(splitter= 'random').fit(x, y)
    clf6 = GaussianMixture(n_components = 2, init_params= 'random').fit(x,y)
    clf7 = GaussianMixture(n_components = 2, init_params= 'random').fit(x,y)
    clf8 = GaussianMixture(n_components = 2, init_params= 'random').fit(x,y)
    clf9 = GaussianNB().fit(x,y)
    clf10 = LogisticRegression(solver='liblinear').fit(x,y)

    classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10]

    # make csv in form of rowNumber, clfNumber, clf prediction on that row
    answers = []
    for point in range(len(df_nolabels)):
        for clf in range(len(classifiers)):
            answers.append([point, clf, classifiers[clf].predict([df_nolabels[point]])])

    count = 0
    f = open("answer_file.csv", "w")
    f.write('question,worker,answer;\n')
    for answer in answers:
        count += 1
        f.write(str(answer[0]) + ',' + str(answer[1]) + ',' + str(int(answer[2]))+'\n')
    f.close()
    p = open("result_file.csv", "w")
    p.close()

    # run VI BP
    subprocess.call(["python", "run.py", "methods/c_EM/method.py", "answer_file.csv", "result_file.csv","decision-making"])

    # extract results, get noisy labels and
    filepath = "result_file.csv"
    noisy_labels = []
    with open(filepath) as fp:
        for line in fp:
            questionAnswer = line.split(',')
            noisy_labels.append(questionAnswer)

    # assign noisy label to proper row
    # combine noisy lables to real labels and randomize
    df_noise_x = []
    df_noise_y = []
    for question in noisy_labels:
        if question[0].rstrip() == 'question':
            continue
        df_noise_x += [df_nolabels[int(question[0].rstrip())]]
        df_noise_y.append(int(question[1].rstrip()))

    count_vi = 0
    for el in range(len(df_noise_x)):
        if label_nolabels[tuple(df_noise_x[el])][0] != df_noise_y[el]:
            count_vi += 1


    df_noise_x += x
    df_noise_y += y
    df_noise_y2 = []
    for el in df_noise_y:
        df_noise_y2.append(int(el))
    df_noise = []
    for el in range(len(df_noise_x)):
        new = df_noise_x[el]
        new.append(df_noise_y2[el])
        df_noise.append(new)

    # need to shuffle the data
    random.shuffle(df_noise)

    df_noise_x = []
    df_noise_y = []
    for row in df_noise:
        df_noise_x.append(row[:-1])
        df_noise_y.append(row[-1:][0])


    # run AdaBoost from Sklearn on noisy data
    bdt2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
    bdt2.fit(df_noise_x, df_noise_y)

    bdt1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
    bdt1.fit(test_x,test_y)

    # Ada boosting on noisy data error rate
    errors = []
    count1 = 0
    for point in df_test:
        est = bdt2.predict([point[:-1]])
        true = int(point[-1:][0])
        est = int(est[0])
        if est == true:
            errors.append([point[:-1],0])
        else:
            count1 += 1
            errors.append([point[:-1],1])

    # Ada boosting on 100 supervised data error rate
    errors = []
    count = 0
    for point in df_test:
        est = bdt1.predict([point[:-1]])
        true = int(point[-1:][0])
        est = int(est[0])
        if est == true:
            errors.append([point[:-1],0])
        else:
            count += 1
            errors.append([point[:-1],1])

    # error rate, noisy -> baseline
    return [count1/len(df_test), count/len(df_test)]


#put to csv file
noisy = []
clean = []
for x in range(100):
    y = run()
    noisy.append(y[0]*100)
    clean.append(y[1]*100)

print(noisy)
print(clean)
