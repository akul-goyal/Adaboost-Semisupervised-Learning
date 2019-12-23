import boosting
def test(clf_list, test):
    count = 0
    for point in test:
        est = boosting.classify_adaBoosting(clf_list, [point[:-1]])
        true = int(point[-1:][0])
        if est != true:
            count += 1
    return (count/len(test))