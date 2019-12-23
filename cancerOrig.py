
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

import warnings

def fxn():
    # In[1218]:


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



    # In[1219]:


    #gets all columns which are not ints and integer encodes them
    obj_df = df.select_dtypes(include=['object']).copy()
    for column in obj_df:
        le = preprocessing.LabelEncoder()
        le.fit(df[column])
        df[column] = le.transform(df[column])


    # In[1220]:


    #normalize all points between [0,1]
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)


    # In[1221]:


    train, test = np.split(df.sample(frac=1), [int(.6*len(df))])
    unlabel, label = np.split(train.sample(frac=1), [int(.8*len(train))])
    test = test.values.tolist()
    nolabels1 = unlabel.values.tolist()
    del unlabel[0]
    nolabels = unlabel.values.tolist()
    labels = label.values.tolist()


    # In[1222]:


    true = {}
    for row in nolabels1:
        true[tuple(row[1:])] = row[:1][0]


    # In[1223]:


    #train f1...fn classifiers on labelled data, will use 6 types: decision stumps, knn, svm, guassian mixture mode
    #                                                               native bayes, logistic regression
    x = []
    y = []
    x_true = []
    y_true = []
    for row in labels:
        x.append(row[1:])
        y.append(row[:1][0])
        x_true.append(row[1:])
        y_true.append(row[:1][0])
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


    # In[1224]:


    # make csv in form of rowNumber, clfNumber, clf prediction on that row
    answers = []
    for point in range(len(nolabels)):
        for clf in range(len(classifiers)):
            answers.append([point, clf, classifiers[clf].predict([nolabels[point]])])

    count = 0
    f = open("answer_file.csv", "w")
    f.write('question,worker,answer;\n')
    for answer in answers:
        count += 1
        f.write(str(answer[0]) + ',' + str(answer[1]) + ',' + str(int(answer[2]))+'\n')
    f.close()
    p = open("result_file.csv", "w")
    p.close()


    # In[1225]:


    #run VI BP
    import subprocess
    subprocess.call(["python", "run.py", "methods/c_EM/method.py", "answer_file.csv", "result_file.csv","decision-making"])


    # In[1226]:


    #extract results, get noisy labels and
    filepath = "result_file.csv"
    noisy_labels = []
    with open(filepath) as fp:
        for line in fp:
            questionAnswer = line.split(',')
            noisy_labels.append(questionAnswer)


    # In[1227]:


    #assign noisy label to proper row
    #combine noisy lables to real labels and randomize
    df_noise_x = []
    df_noise_y = []
    for question in noisy_labels:
        if question[0].rstrip() == 'question':
            continue
        df_noise_x += [nolabels[int(question[0].rstrip())]]
        df_noise_y.append(int(question[1].rstrip()))


    count_vi = 0
    for el in range(len(df_noise_x)):
        if true[tuple(df_noise_x[el])] != df_noise_y[el]:
            count_vi += 1


    # In[1228]:


    df_noise_x += x
    df_noise_y += y

    df_noise = []
    for el in range(len(df_noise_x)):
        new = df_noise_x[el]
        new.append(df_noise_y[el])
        df_noise.append(new)

    #need to shuffle the data
    random.shuffle(df_noise)

    df_noise_x = []
    df_noise_y = []
    for row in df_noise:
        df_noise_x.append(row[:-1])
        df_noise_y.append(row[-1:][0])


    # In[1229]:


    #run AdaBoost from Sklearn on noisy data
    bdt2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME",
                             n_estimators=20)
    bdt2.fit(df_noise_x, df_noise_y)


    # In[1230]:


    bdt1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             algorithm="SAMME",
                             n_estimators=20)
    bdt1.fit(x_true,y_true)


    # In[1231]:


    #Ada boosting on noisy data error rate
    errors = []
    count1 = 0
    for point in test:
        est = bdt2.predict([point[1:]])
        true = int(point[0])
        est = int(est[0])
        if est == true:
            errors.append([point[:-1],0])
        else:
            count1 += 1
            errors.append([point[:-1],1])

    #Ada boosting on 500 supervised data error rate
    errors = []
    count = 0
    for point in test:
        est = bdt1.predict([point[:-1]])
        true = int(point[-1:][0])
        est = int(est[0])
        if est == true:
            errors.append([point[:-1],0])
        else:
            count += 1
            errors.append([point[:-1],1])

    #error rate, noisy -> baseline
    #print('\nBreast-cancer: error of VI-BP: ',count_vi, ' Out of: ', len(df_noise_x), ' Error on Noisy Ada Boosting: ',count1,' Error on Norm Ada Boosting: ', count, ' Error rate on Noisy Ada Boosting: ', count1/len(test), ' Error on Norm Ada Boosting: ',count/len(test))

    return (count1/len(test),count/len(test))

# In[ ]:
avg1 = 0
avg2 = 0
for x in range(50):
    ret = fxn()
    avg1 += ret[0]
    avg2 += ret[1]
avg1 = avg1/50
avg2 = avg2/50
print(avg1,avg2)