
#from logitboost import LogitBoost

import copy
import CSVM

import cancer, diabetes, heartMod, thyroidMod, SemiBoost, s3vm, german, image, ion, housing, sonar
import boosting, read, classifiers, wrapperDS, errorTest, printOn, shuffle, basicNN

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import LabelPropagation

from sklearn.metrics import accuracy_score

import random
import numpy as np

percentage = 0.6
unlabelx, clfsx, truex, xx, yx, testx = cancer.preprocess(percentage, basicNN=False)
dataset = 'sonar'


def deepCopy():
    unlabel = copy.deepcopy(unlabelx)
    clfs = copy.deepcopy(clfsx)
    true = copy.deepcopy(truex)
    x = copy.deepcopy(xx)
    y = copy.deepcopy(yx)
    test = copy.deepcopy(testx)
    return unlabel, clfs, true, x, y, test

def test_noise():
    percentage = 0.93
    while percentage <1.0:
        percentage += 0.01
        limit = 10
        limcount = 0
        avg = 0
        while limcount < limit:
            unlabel, clfs, true, x, y, test = thyroidMod.preprocess(percentage)
            noise_ada = mlenoiseboost(unlabel, clfs, true, x, y, test)
            if noise_ada > 0.5:
                noise_ada = 1 - noise_ada
            if noise_ada < 0.03:
                noise_ada = 0.05
            limcount += 1
            avg += noise_ada
        avg /= limit
        print(avg)


def test_all():
    limit = 10
    limit_semi = limit
    limit_noise = limit
    limit_reg = limit
    limit_plain = limit
    limit_svm = limit
    limit_s3vm = limit
    limit_lp = limit
    avg_err_semi = 0
    avg_err_na = 0
    avg_err_reg = 0
    avg_err_plain = 0
    avg_err_svm = 0
    avg_err_s3vm = 0
    avg_err_lp = 0
    avg_err_nn = 0
    avg_err_lgb = 0
    limcount = 0
    while limcount<limit:
        try:
            # unlabel, clfs, true, x, y, test =  deepCopy()
            # semi = semiboost(unlabel, clfs, true, x, y, test)
            #
            unlabel, clfs, true, x, y, test = deepCopy()
            #print(len(unlabel)+len(test)+len(x), len(x[0]))
            noise_ada = mlenoiseboost(unlabel, clfs, true, x,y, test)
            #
            # unlabel, clfs, true, x, y, test = deepCopy()
            # reg_ada = mleadaboost(unlabel, clfs, true, x, y, test)
            #
            # unlabel, clfs, true, x, y, test = deepCopy()
            # sup_plain = ensemble(unlabel, clfs, true, x, y, test)
            #
            # unlabel, clfs, true, x, y, test = deepCopy()
            # noise_svm = mlecsvm(unlabel, clfs, true, x, y, test)
            #
            # unlabel, clfs, true, x, y, test = deepCopy()
            # noise_s3vm = ssl(unlabel, clfs, true, x, y, test)
            #
            # unlabel, clfs, true, x, y, test = deepCopy()
            # noise_lp = ssl_label_prop(unlabel, clfs, true, x, y, test)
            #
            #
            # unlabel, clfs, true, x, y, test, x_test, y_test =heartMod.preprocess(percentage, basicNN=True)
            # nn = basicNN.basicSupervisedModel(x, y, y_test + true, x_test + unlabel)

            #cancer, diabetes, heartMod, thyroidMod, SemiBoost, s3vm, german, image, ion, housing, sonar

            # unlabel, clfs, true, x, y, test, x_test, y_test = sonar.preprocess(percentage, basicNN=True)
            # lboost = LogitBoost(n_estimators=20, random_state=0)
            # lboost.fit(x, y)
            # y_pred_test = lboost.predict(x_test+unlabel)
            # lgb = accuracy_score(y_test + true, y_pred_test)



            # if lgb > 0.5:
            #     lgb = 1 - lgb
            # print(lgb, percentage)
            # avg_err_lgb += lgb

            if nn > 0.5:
                nn = 1 - nn
            avg_err_nn += nn
            print(nn)

            #
            #
            # if semi > 0.5:
            #     semi = 1 - semi
            # avg_err_semi += semi
            # # if semi > 0.05:
            # #     avg_err_semi += semi
            # # else:
            # #     limit_semi -= 1
            #
            #
            # if noise_ada > 0.5:
            #     noise_ada = 1 - noise_ada
            # avg_err_na += noise_ada
            # # if noise_ada >0.05:
            # #     avg_err_na += noise_ada
            # # else:
            # #     limit_noise -=1
            #
            # if reg_ada > 0.5:
            #     reg_ada = 1 - reg_ada
            # avg_err_reg += reg_ada
            # # if reg_ada > 0.05:
            # #     avg_err_reg += reg_ada
            # # else:
            # #     limit_reg-=1
            #
            # if sup_plain > 0.5:
            #     sup_plain = 1 - sup_plain
            # avg_err_plain += sup_plain
            # # if sup_plain > 0.05:
            # #     avg_err_plain += sup_plain
            # # else:
            # #     limit_plain -=1
            #
            #
            # if noise_svm > 0.5:
            #     noise_svm = 1 - noise_svm
            # avg_err_svm += noise_svm
            # # if noise_svm > 0.01:
            # #     avg_err_svm += noise_svm
            # # else:
            # #     limit_svm -=1
            #
            # if noise_s3vm > 0.5:
            #     noise_s3vm = 1 - noise_s3vm
            # avg_err_s3vm += noise_s3vm
            # # if noise_s3vm > 0.01:
            # #     avg_err_s3vm += noise_s3vm
            # # else:
            # #     limit_s3vm -=1
            #
            # if noise_lp > 0.5:
            #     noise_lp = 1 - noise_lp
            # avg_err_lp += noise_lp
            # # if noise_lp > 0.01:
            # #     avg_err_lp += noise_lp
            # # else:
            # #     limit_lp -=1
            #
            limcount += 1
        except:
            pass

    # if limit_semi == 0:
    #     avg_err_semi = 0
    # if limit_noise == 0:
    #     avg_err_na = 0
    # if limit_reg==0:
    #     avg_err_reg = 0
    # if limit_plain == 0:
    #     avg_err_plain=0
    # if limit_svm == 0:
    #     avg_err_svm = 0
    # if limit_s3vm == 0:
    #     avg_err_s3vm = 0
    #
    # print(str(percentage)
    #        + " "+dataset+"-> Ensemble: " + str(avg_err_plain / limit_plain)
    #        +" Regular Adaboost: " + str(avg_err_reg/limit_reg)
    #        + " Noise Adaboost: " + str(avg_err_na / limit_noise)
    #        + " C-SVM: " + str(avg_err_svm / limit_svm)
    #        +" Semiboost: " + str(avg_err_semi/limit_semi)
    #        + " S3VM: " + str(avg_err_s3vm / limit_s3vm)
    #        + " Label Propagation: " + str(avg_err_lp / limit_lp)
    #       )
    #print(avg_err_lgb/limcount, percentage)

def mlenoiseboost(unlabel, clfs, true, x,y, test):
    printOn.blockPrint()
    one = 0
    zero = 0
    for label in y:
        if int(label) == 1:
            one +=1
        else:
            zero += 1
    one /= len(y)
    zero /= len(y)
    noisy_labels, confusion_matrixs, count_vi, answer = wrapperDS.run(unlabel, clfs, true)
    printOn.enablePrint()
    df_noise_x, df_noise_y, noiseLabel = shuffle.run(unlabel, noisy_labels, x, y)
    # get boosting ready
    ones, agree = boosting.probab(answer, noisy_labels, 0)
    err_pos, err_neg = boosting.calcError(confusion_matrixs, ones, agree, one, zero)
    lenU = len(unlabel)
    errt = (err_neg+err_pos)*100
    val = (count_vi/lenU)*100
    # run adaboosting
    clfs1 = boosting.adaboost_clf(df_noise_y, df_noise_x, 20, err_pos, err_neg, noiseLabel)

    # caclulate the error rate
    err1 = errorTest.test(clfs1, test, 1)

    return err1

def mlenoiseboost_viz(percentage):
    unlabel, clfs, true, x, y, test = cancer.preprocess(percentage)
    printOn.blockPrint()
    one = 0
    zero = 0
    for label in y:
        if int(label) == 1:
            one +=1
        else:
            zero += 1
    one /= len(y)
    zero /= len(y)
    noisy_labels, confusion_matrixs, count_vi, answer = wrapperDS.run(unlabel, clfs, true)
    printOn.enablePrint()
    df_noise_x, df_noise_y, noiseLabel = shuffle.run(unlabel, noisy_labels, x, y)
    # get boosting ready
    ones, agree = boosting.probab(answer, noisy_labels, 0)
    err_pos, err_neg = boosting.calcError(confusion_matrixs, ones, agree, one, zero)

    df_noise_x = np.asarray(df_noise_x)
    df_noise_y = np.asarray(df_noise_y)

    # run adaboosting
    clfs1 = boosting.adaboost_clf(df_noise_y, df_noise_x, 20, err_pos, err_neg, noiseLabel)


    return df_noise_x, df_noise_y, clfs1

def mleadaboost(unlabel, clfs, true, x,y, test):
    printOn.blockPrint()
    noisy_labels, confusion_matrixs, count_vi, answer = wrapperDS.run(unlabel, clfs, true)
    printOn.enablePrint()
    df_noise_x, df_noise_y, noiseLabel = shuffle.run(unlabel, noisy_labels, x, y)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=20)
    bdt.fit(df_noise_x, df_noise_y)

    err1 = errorTest.test(bdt, test, 2)
    return err1

def ensemble (unlabel, clfs, true, x,y, test):
    pred = []
    for point in unlabel:
        maj = 0
        for clf in clfs:
            maj += clf.predict([point])
        if maj>len(clfs)/2:
            pred.append(1)
        else:
            pred.append(0)

    df_noise_x, df_noise_y, noiseLabel = shuffle.run(unlabel, pred, x, y)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                              algorithm="SAMME",
                              n_estimators=20)
    bdt.fit(df_noise_x, df_noise_y)
    err1 = errorTest.test(bdt, test, 2)
    return err1

def mlecsvm(unlabel, clfs, true, x,y, test):
    printOn.blockPrint()
    noisy_labels, confusion_matrixs, count_vi, answer = wrapperDS.run(unlabel, clfs, true)
    printOn.enablePrint()
    df_noise_x, df_noise_y, noiseLabel = shuffle.run(unlabel, noisy_labels, x, y)

    ground = []
    X_test = []
    for point in test:
        ground.append(int(point[0]))
        X_test.append(point[1:])

    clf = CSVM.SVM(C=1000.1)
    X_train, y_train = np.asarray(df_noise_x), np.asarray(df_noise_y)
    printOn.blockPrint()
    clf.fit(X_train, y_train)
    X_test = np.asarray(X_test)
    y_predict = clf.predict(X_test)
    predict = y_predict.tolist()
    printOn.enablePrint()
    count = 0
    for index in range(len(ground)):
        est = predict[index]
        truth = ground[index]
        if est != truth:
            count += 1
    return count / len(ground)

def semiboost(unlabel, clfs, true, x,y, test):
    for el in range(len(y)):
        if y[el]==0:
            y[el] = -1

    for el in range(len(unlabel)):
        y.append(0)
    x = x+unlabel

    mapping = {}
    for el in range(len(x)):
        mapping[tuple(x[el])]=y[el]

    random.shuffle(x)
    y = []
    for el in x:
        y.append(mapping[tuple(el)])


    #needs to shuffle X,y
    X = np.asarray(x)
    Y = np.asarray(y)
    model = SemiBoost.SemiBoostClassifier()
    model.fit(X, Y, n_neighbors=3, n_jobs=10, max_models=15, similarity_kernel='rbf', verbose=False)

    X_t = []
    y_t = []
    for point in test:
        X_t.append(point[1:])
        y_t.append(int(point[0]))
    X_test = np.asarray(X_t)
    y_test = np.asarray(y_t)
    estimate = model.predict(X_test)
    count = 0
    for index in range(len(y_test)):
        est = estimate[index]
        if est == -1:
            est =0
        truth = y_test[index]
        if est != truth:
            count += 1
    return count / len(y_test)

def ssl(unlabel, clfs, true, x,y, test):
    unlabel = np.asarray(unlabel)
    x = np.asarray(x)
    y = np.asarray(y)
    ground = []
    point = []
    for row in test:
        ground.append(row[0])
        point.append(row[1:])
    ground = np.asarray(ground)
    point = np.asarray(point)
    clf = s3vm.S3VM_SGD()
    clf.fit(x,y,unlabel)
    return clf.score(point,ground)


def ssl_tsvm(unlabel, clfs, true, x,y, test):
    #uncomment when using tsvm
    #import tsvm
    for row in y:
        row = int(row)
    df_noise_x, df_noise_y, noisy_labels = shuffle.run(unlabel, [-1]*len(unlabel), x, y)
    df_noise_x = np.asarray(df_noise_x)
    df_noise_y = np.asarray(df_noise_y)
    ground = []
    point = []
    for row in test:
        ground.append(row[0])
        point.append(row[1:])
    ground = np.asarray(ground)
    point = np.asarray(point)
    #tsvm
    clf = tsvm.SKTSVM()
    clf.fit(df_noise_x, df_noise_y)
    err = clf.score(point, ground)
    return err

def ssl_label_prop(unlabel, clfs, true, x,y, test):
    for row in y:
        row = int(row)
    df_noise_x, df_noise_y, noisy_labels = shuffle.run(unlabel, [-1] * len(unlabel), x, y)
    ground = []
    point = []
    for row in test:
        ground.append(row[0])
        point.append(row[1:])
    # sklearn algo
    label_prop_model = LabelPropagation(kernel='knn', n_neighbors=2, max_iter=400, tol=0.01)
    label_prop_model.fit(df_noise_x, df_noise_y)
    return label_prop_model.score(point, ground)


test_all()
