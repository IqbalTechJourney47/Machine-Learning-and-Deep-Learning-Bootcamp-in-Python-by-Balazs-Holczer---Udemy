# Multiclass classification implementation

import numpy as np
from keras.models import Sequential
# from keras.layers.core import Dense
# from tensorflow.keras.layers import Dense
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
# from tensorflow.keras.optimizers import Adam

dataset = load_iris()

features = dataset.data
print (features)

y = dataset.target.reshape(-1, 1)

# 1D array without reshape
print(dataset.target)

# 2D array with reshape
print(y)

encoder = OneHotEncoder(sparse=False)
targets = encoder.fit_transform(y)

# If we do not do something like this (sparse=False), then python is not going to transform the target variables into a
# one hot encoded vector as we desire

# One hot encoding
# It means we are going to represent data in 1-D array

train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.2)

model = Sequential()

# first parameter is output dimension
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# 3 output neurons
# model.add(Dense(3, activation='softmax'))

# we can define the loss function MSE or negative log likelihood
# optimizer will find the right adjustments for the weights: SGD, Adagrad, ADAM ...

# optimizer = Adam(lr=0.005)   # lr = learning rate
learning_rate = 0.005
optimizer = Adam(learning_rate=learning_rate)

model.compile(loss = 'categorical_crossentropy',
             optimizer = optimizer,
             metrics = 'accuracy')

# Mean Squared Error (MSE)
# We use mean squared error when we are dealing with regression(so we have only one output feature)
# Negative Log Likelihood
# We use negative log likelihood function when dealing with classification

model.fit(train_features, train_targets, epochs=1000, batch_size=20, verbose=2)

