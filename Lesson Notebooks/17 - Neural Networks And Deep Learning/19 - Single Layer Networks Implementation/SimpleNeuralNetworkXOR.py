from keras.models import Sequential
# from keras.layers.core import Dense
from tensorflow.keras.layers import Dense
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print(X.shape)

print(X)

print(y)

model = Sequential()

# neurons = 4
# input layer is going to be defined with the help of input_dim
model.add(Dense(4, input_dim=2, activation='sigmoid'))   # hidden layer
model.add(Dense(1, input_dim=4, activation='sigmoid'))   # output layer

print(model.weights)

# Compilation
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

model.fit(X, y, epochs=10000, verbose=2)

print('Predictions after the training...')
print(model.predict(X))
