from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm

#train f1...fn classifiers on labelled data, will use 6 types: decision stumps, knn, svm, guassian mixture mode
    #                                                               native bayes, logistic regression
def ensemble(x, y):
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
    return classifiers