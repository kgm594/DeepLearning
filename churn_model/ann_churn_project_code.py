import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint

data_address = '/home/ariya/Desktop/codes/artificial_neural_networks/project1/Churn_Modelling.csv'

# read the data from the file
dataset = pd.read_csv(data_address)
# split the relevant data
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# encode the string categories
encode_x_1 = LabelEncoder()
X[:, 1] = encode_x_1.fit_transform(X[:, 1])
encode_x_2 = LabelEncoder()
X[:, 2] = encode_x_2.fit_transform(X[:, 2])

# do the categorical splitting for the countries
categorical_split = OneHotEncoder(categorical_features=[1])
X = categorical_split.fit_transform(X).toarray()

# bypass the dummy variable trap due to categorical encoding of countries
X = X[:, 1:]

# edit the data for cross validation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# scaling the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# initialize the ANN
classifier = Sequential()

# add the first hidden layer
classifier.add(Dense(units=23, kernel_initializer='uniform', activation='tanh', input_dim=11))
# add the second hidden layer
classifier.add(Dense(units=13, kernel_initializer='uniform', activation='sigmoid'))
# add the third hidden layer
classifier.add(Dense(units=7, kernel_initializer='uniform', activation='relu'))
# add the forth hidden layer
classifier.add(Dense(units=4, kernel_initializer='uniform', activation='sigmoid'))
# add the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# set up the compiler
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# save the networks
# filepath="weights.best.hdf5"
# filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

# save the model to a file
# model_json = classifier.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

# fit the data
# classifier.fit(X_train, Y_train, validation_split=0.15, batch_size=1, nb_epoch=100, callbacks=callbacks_list, verbose=0)
classifier.fit(X_train, Y_train, batch_size=1, nb_epoch=100)

# run the network on test data
Y_predict = classifier.predict(X_test)
Y_predict = (Y_predict > 0.5)

# check how many of the test data is predicted correctly
cm = confusion_matrix(Y_test, Y_predict)
print('accuracy on the test set: ', (cm[0][0] + cm[1][1])/(cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]))


