#!/usr/bin/env python
# coding: utf-8

# In[587]:


import pandas as pd
import numpy as np
from matplotlib import style

from sklearn import preprocessing

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.neural_network import BernoulliRBM
from sklearn.mixture import BayesianGaussianMixture
from sklearn.manifold import Isomap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.ensemble import AdaBoostClassifier
import random


# In[588]:

def fxn():
    #read in the data
    df = pd.read_csv('data.csv')

    #columns to drop
    df = df.drop(['id'], axis=1)

    df.sample(frac=1)
    #gets rid of ? and one hot encoding for all columns that need it
    index = []
    count = 0
    for val in range(len(df.ix[:,0])):
        flag = False
        for column in df:
            if df[column][val] == '?':
                flag = True
                break
        if flag:
            continue
        if count<1000:
            index.append(val)
            count += 1
    df = df[df.index.isin(index)]


    #gets all columns which are not ints and integer encodes them
    obj_df = df.select_dtypes(include=['object']).copy()
    for column in obj_df:
        le = preprocessing.LabelEncoder()
        le.fit(df[column])
        df[column] = le.transform(df[column])

    #normalize all points between [0,1]
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)


    # In[589]:


    #make dataset only 1100
    #create 500/500 split between labelled on nonlablled array, 1000 semi-sup data set, and 100 validation dataset
    train, test = np.split(df.sample(frac=1), [int(.8*len(df))])
    #print(train)
    train = train.values.tolist()
    test = test.values.tolist()

    df_unsupervised = []

    label_nolabels = {}
    for point in train:
        #unlablled 1000 points data
        df_unsupervised.append(point[1:])
        label_nolabels[tuple(point[1:])]= [point[0]]


    # In[590]:


    ##### #kmeans_forest 1-10, unsupervised learning adaboosting
    # kmeans1 = KMeans(n_clusters=2).fit(df_unsupervised)
    # # #kmeans2 = SpectralClustering(n_clusters = 2).fit_predict(df_unsupervised).tolist()
    # # kmeans3 = MeanShift().fit(df_unsupervised)
    # # #kmeans4 = AgglomerativeClustering(n_clusters=2).fit_predict(df_unsupervised).tolist()
    # # kmeans5 = DBSCAN().fit_predict(df_unsupervised).tolist()
    # # kmeans6 = GaussianMixture(n_components=2).fit(df_unsupervised)
    # # kmeans7 = Birch(n_clusters=2).fit(df_unsupervised)
    # # kmeans8 = BayesianGaussianMixture(n_components=2).fit(df_unsupervised)
    # classifiers = [kmeans1, kmeans3, kmeans5, kmeans6, kmeans7, kmeans8]
    #kmeans_forest 1-10, unsupervised learning adaboosting
    kmeans1 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans2 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans3 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans4 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans5 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans6 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans7 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans8 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans9 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans10 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans11 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans12 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans13 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans14 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans15 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans16 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans17 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans18 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans19 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans20 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans21 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans22 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans23 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans24 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans25 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans26 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans27 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans28 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans29 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans30 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans31 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans32 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans33 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans34 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans35 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans36 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans37 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans38 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans39 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans40 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans41 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans42 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans43 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans44 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans45 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans46 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans47 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans48 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans49 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    kmeans50 = KMeans(n_clusters=2, init='random', n_init=10).fit(np.asarray(df_unsupervised))
    classifiers = [kmeans1,kmeans2,kmeans3,kmeans4,kmeans5,kmeans6,kmeans7,kmeans8,kmeans9,kmeans10,              kmeans11,kmeans12,kmeans13,kmeans14,kmeans15,kmeans16,kmeans17,kmeans18,kmeans19,kmeans20,              kmeans21,kmeans22,kmeans23,kmeans24,kmeans25,kmeans26,kmeans27,kmeans28,kmeans29,kmeans30,              kmeans31,kmeans32,kmeans33,kmeans34,kmeans35,kmeans36,kmeans37,kmeans38,kmeans39,kmeans40,              kmeans41,kmeans42,kmeans43,kmeans44,kmeans45,kmeans46,kmeans47,kmeans48,kmeans49,kmeans50]


    # In[591]:


    # make csv in form of rowNumber, clfNumber, clf prediction on that row
    answers = []
    for point in range(len(df_unsupervised)):
        for clf in range(len(classifiers)):
            answers.append([point, clf, classifiers[clf].predict([df_unsupervised[point]])])

    count = 0
    f = open("answer_file.csv", "w")
    f.write('question,worker,answer;\n')
    for answer in answers:
        count += 1
        f.write(str(answer[0]) + ',' + str(answer[1]) + ',' + str(int(answer[2]))+'\n')
    f.close()
    p = open("result_file.csv", "w")
    p.close()


    # In[592]:


    #run VI BP
    import subprocess
    subprocess.call(["python", "run.py", "methods/c_EM/method.py", "answer_file.csv", "result_file.csv","decision-making"])


    # In[593]:


    #extract results, get noisy labels and
    filepath = "result_file.csv"
    noisy_labels = []
    with open(filepath) as fp:
        for line in fp:
            questionAnswer = line.split(',')
            noisy_labels.append(questionAnswer)


    # In[594]:


    #assign noisy label to proper row
    df_noise_x = []
    df_noise_y = []
    for question in noisy_labels:
        if question[0].rstrip() == 'question':
            continue
        df_noise_x += [df_unsupervised[int(question[0].rstrip())]]
        df_noise_y.append(int(question[1].rstrip()))
    count_vi = 0
    for el in range(len(df_noise_x)):
        if label_nolabels[tuple(df_noise_x[el])][0] != df_noise_y[el]:
            count_vi += 1
    print(count_vi,len(df_noise_x))


    # In[595]:


    df_noise_y2 = []
    for el in df_noise_y:
        df_noise_y2.append(int(el))

    df_noise = []
    for el in range(len(df_noise_x)):
        new = df_noise_x[el]
        new.append(df_noise_y2[el])
        df_noise.append(new)

    #need to shuffle the data
    random.shuffle(df_noise)

    df_noise_x = []
    df_noise_y = []
    for row in df_noise:
        df_noise_x.append(row[:-1])
        df_noise_y.append(row[-1:][0])


    # In[596]:


    #run AdaBoost from Sklearn on noisy data
    bdt2 = AdaBoostClassifier(DecisionTreeClassifier(),
                             algorithm="SAMME",
                             n_estimators=20)
    bdt2.fit(df_noise_x, df_noise_y)


    # In[597]:


    #Ada boosting on noisy data error rate
    errors = []
    count1 = 0
    for point in test:
        est = bdt2.predict([point[:-1]])
        true = int(point[-1:][0])
        est = int(est[0])
        if est == true:
            errors.append([point[:-1],0])
        else:
            count1 += 1
            errors.append([point[:-1],1])

            # error rate, noisy -> baseline
    return (count1 / len(test))

avg1 = 0
avg2 = 0
for x in range(30):
    ret = fxn()
    if ret > .50:
        ret = 1 - ret
    avg1 += ret
avg1 = avg1 / 30
print(avg1)