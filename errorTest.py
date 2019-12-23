import boosting
def test(clf_list, test, tp):
    count = 0
    for point in test:
        if tp == 1:
            est = boosting.classify_adaBoosting(clf_list, [point[1:]])
        else:
            est = int(clf_list.predict([point[1:]]))
        true = int(point[0])
        if est != true:
            count += 1
    return (count/len(test))
