import random
def run(unlabel, noisy_labels, x, y):
    mapping = {}
    for row in unlabel:
        mapping[tuple(row)]=1
    for row in x:
        mapping[tuple(row)]=0

    unlabel += x
    noisy_labels += y


    df_noise = []
    for el in range(len(unlabel)):
        new = unlabel[el]
        new.append(noisy_labels[el])
        df_noise.append(new)

    #need to shuffle the data
    random.shuffle(df_noise)

    df_noise_x = []
    df_noise_y = []
    for row in df_noise:
        df_noise_x.append(row[:-1])
        df_noise_y.append(row[-1:][0])

    noisy_labels=[]
    for row in df_noise_x:
        noisy_labels.append(mapping[tuple(row)])
    return df_noise_x, df_noise_y, noisy_labels