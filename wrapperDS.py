# Answers in format {patients: {observers: [labels]}}
import dawid_skene
def run (unlabel, clfs, true):
    answer = {}
    for point in range(len(unlabel)):
        answer[point] = {}
        for clf in range(len(clfs)):
            answer[point][clf]=[int(clfs[clf].predict([unlabel[point]]))]
    patient_class, error_rates = dawid_skene.main(answer)
    noisy_labels = []
    for i in range(len(unlabel)):
        if patient_class[i,:][0] > patient_class[i,:][1]:
            noisy_labels.append(0)
        else:
            noisy_labels.append(1)

    count_vi = 0
    for el in range(len(unlabel)):
        if true[tuple(unlabel[el])] != noisy_labels[el]:
            count_vi += 1
    return noisy_labels, error_rates, count_vi, answer
