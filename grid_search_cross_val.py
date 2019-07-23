import pandas as pd 
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# import the scale data set

x_train = pd.read_csv('x_train_scale.csv',index_col=0)
x_test = pd.read_csv('x_test_scale.csv',index_col=0)
y_train = pd.read_csv('y_train.csv',index_col=0)
y_test = pd.read_csv('y_test.csv',index_col=0)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Importing the keras libraries and package

import keras 
from keras.models import Sequential
from keras.layers import Dense

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

######################################################################
def build_classifier(optimizer):
    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 19, kernel_initializer = 'uniform',
                     activation = 'relu', input_dim = 35))

    # Adding the second hidden layer 
    classifier.add(Dense(units=19, kernel_initializer='uniform',
                     activation='relu'))

    # Adding the third hidden layer 
    classifier.add(Dense(units=19, kernel_initializer='uniform',
                     activation='relu'))

    '''# Adding the fourth hidden layer 
    classifier.add(Dense(units=19, kernel_initializer='uniform',
                     activation='relu'))'''


    # Adding the output layer 
    classifier.add(Dense(units=1, kernel_initializer='uniform',
                     activation='sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
    
    return classifier


###################################################################
    
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [50,100,200],
              'epochs': [5, 10, 15],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_