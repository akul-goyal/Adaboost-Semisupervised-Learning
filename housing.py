import boosting, read, classifiers, wrapperDS, errorTest, printOn, shuffle
def preprocess(percentage, basicNN = False):
    printOn.blockPrint()
    if basicNN == True:
        test, unlabel, label, true, x, y, x_true, y_true, x_test,y_test = read.read(file='housing.data', drop=None, retNum=1, chopNum=1,
                                                                 unlabel_percentage=percentage, transform=True, ytrain=True)
    else:
        test, unlabel, label, true, x, y, x_true, y_true = read.read(file='housing.data', drop=None, retNum=1, chopNum=1,
                                                                 unlabel_percentage=percentage, transform=True)
    clfs = classifiers.ensemble(x,y)
    printOn.enablePrint()
    for point in test:
        point.insert(0, point.pop())
    if basicNN == True:
        return unlabel, clfs, true, x,y, test, y_test, x_test
    else:
        return unlabel, clfs, true, x, y, test