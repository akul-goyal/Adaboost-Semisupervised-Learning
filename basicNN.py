import tensorflow as tf
import pandas as pd
import numpy as np
from keras import metrics

#Using Keras Sequentail Model, this is bound to change
def basicSupervisedModel(x_train,y_train,y_test, x_test):
    model = tf.keras.models.Sequential()
    #Using 3 hidden layers with 128 nodes relu activation function, bound to change
    #model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100,activation='sigmoid'))
    model.add(tf.keras.layers.Dense(100, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
    #using adam optimizer, and MSE
    model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])
    #training model for 3 epochs, will increase for better accuracy
    for index in range(len(x_train)):
        x_train[index] = [x_train[index]]
    for index in range(len(x_test)):
        x_test[index] = [x_test[index]]
    with open('listfile2.txt', 'w') as filehandle:
        for listitem in x_test:
            filehandle.write('%s\n' % listitem)
    with open('listfile3.txt', 'w') as filehandle:
        for listitem in y_test:
            filehandle.write('%s\n' % listitem)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    model.fit(x_train,y_train,epochs=15) #validate model
    loss, accuracy = model.evaluate(x_test,y_test)
    return accuracy