import pandas as pd
import numpy as np

from sklearn import preprocessing

import random

#read in the data
def read(file, drop, retNum, chopNum, point_percentage = 1, train_percentage = 0.6, unlabel_percentage = 0.9, multiclass = False, transform=False, ytrain=False):
    df = pd.read_csv(file)
    df = df.sample(frac=1)
    #columns to drop
    if drop != None:
        df = df.drop(drop, axis=1)

    df = df.sample(frac=1)
    #gets rid of ? and one hot encoding for all columns that need it
    index = []
    length = (len(df.index)-1)
    for val in range(length):
        flag = False
        for column in df:
            if df[column][val] == '?':
                flag = True
                break
        if flag:
            continue
        index.append(val)
    if transform:
        df['label'] = np.where(df['label'] > (df['label'].max()+df['label'].min())/2, 1, 0)

    #gets all columns which are not ints and integer encodes them
    obj_df = df.select_dtypes(include=['object']).copy()
    for column in obj_df:
        le = preprocessing.LabelEncoder()
        le.fit(df[column])
        df[column] = le.transform(df[column])

    if multiclass:
        label = df['label'].values
        df = df.drop(['label'], axis=1)
    # normalize all points between [0,1]
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    if multiclass:
        df['label']= label

    #split the data into train and test
    train, test = np.split(df.sample(frac=1), [int(train_percentage*len(df))])
    #split the data in unlabelled and labeled
    unlabel, label = np.split(train.sample(frac=1), [int(unlabel_percentage*len(train))])
    test = test.values.tolist()
    labels = label.values.tolist()
    temp = unlabel.values.tolist()
    unlabel = []

    #map the unlabelled data to their ground truth
    if ytrain==False:
        ground_truth = {}
    else:
        ground_truth = []
    if chopNum == 0:
        for row in temp:
            if ytrain == False:
                ground_truth[tuple(row[1:])] = row[0]
            else:
                ground_truth.append(row[0])
            unlabel.append(row[1:])
    if chopNum == 1:
        for row in temp:
            if ytrain == False:
                ground_truth[tuple(row[:-1])] = row[-1:][0]
            else:
                ground_truth.append(row[-1:][0])
            unlabel.append(row[:-1])
    if chopNum == 2:
        train = train.values.tolist()
        for row in temp:
            if row[0] > 0:
                if ytrain == False:
                    ground_truth[tuple(row[1:])] = 1
                else:
                    ground_truth.append(1)
            else:
                if ytrain == False:
                    ground_truth[tuple(row[1:])] = 0
                else:
                    ground_truth.append(0)
            unlabel.append(row[1:])
        for row in test:
            if row[0]>0:
                row[0]=1
        for row in train:
            if row[0]>0:
                row[0]=1
        train = np.asarray(train)
    if retNum == 0:
        return train,test,unlabel,labels, ground_truth, temp
    if retNum == 1:
        if chopNum == 0:
            x = []
            y = []
            x_true = []
            y_true = []
            for row in labels:
                x.append(row[1:])
                y.append(row[:1][0])
                x_true.append(row[1:])
                y_true.append(row[:1][0])
            y_test= []
            x_test = []
            for row in test:
                x_test.append(row[1:])
                y_test.append(row[:1][0])
        if chopNum == 1:
            x = []
            y = []
            x_true = []
            y_true = []
            for row in labels:
                x.append(row[:-1])
                y.append(row[-1:][0])
                x_true.append(row[:-1])
                y_true.append(row[-1:][0])
            x_test = []
            y_test = []
            for row in test:
                x_test.append(row[:-1])
                y_test.append(row[-1:][0])
        if chopNum == 2:
            x = []
            y = []
            x_true = []
            y_true = []
            for row in labels:
                if row[:1][0] > 0:
                    x.append(row[1:])
                    y.append(1)
                    x_true.append(row[1:])
                    y_true.append(1)
                else:
                    x.append(row[1:])
                    y.append(0)
                    x_true.append(row[1:])
                    y_true.append(0)
            x_test = []
            y_test = []
            for row in test:
                if row[:1][0] > 0:
                    x_test.append(row[1:])
                    y_test.append(1)
                else:
                    x_test.append(row[1:])
                    y_test.append(0)

        if ytrain == False:
            return test,unlabel,labels, ground_truth, x, y, x_true, y_true
        else:
            return test, unlabel, labels, ground_truth, x, y, x_true, y_true, y_test, x_test