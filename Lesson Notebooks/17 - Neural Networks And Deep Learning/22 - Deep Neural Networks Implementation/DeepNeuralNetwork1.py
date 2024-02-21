# Deep neural network implementation

import numpy as np
from keras.models import Sequential
# from keras.layers.core import Dense
from tensorflow.keras.layers import Dense

# Why XOR? Because it is a non-linealy separable problem
# XOR problem training samples

training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], 'float32')

# XOR problem target values accordingly
target_data = np.array([[0], [1], [1], [0]], 'float32')

# we can define the neural network layers in a sequential manner
model = Sequential()
# first parameter is output dimension
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# we can define the loss function MSE or negative log likelihood
# optimizer will find the right adjustments for the weights: SGD, Adagrad, ADAM ...
model.compile(loss='mean_squared_error',
             optimizer='adam',
             metrics=['binary_accuracy'])



# activation='sigmoid'
# If we use sigmoid function, then because of the vanishing gradient, which means that the derivative of the sigmoid
# may be so small that the update operations are not going to work that much,
# which means that the algorithm needs more epochs

# loss='mean_squared_error'
# loss function value is going to decrease during the training procedure.
# So at the beginning, there is a high error, which means that the value of the loss function is high as well.
# And during the training procedure, the algorithm keeps updating the weights, which means that the model keeps making
# better and better predictions, which means that the error will decrease
# And of course, because the error decreases, the loss function value will decrease as well

# epoch is an iteration over the entire dataset
# verbose 0 is silent 1 and 2 are showing results
model.fit(training_data, target_data, epochs=500, verbose=2)

# epoch is single iteration over the entire dataset

# of course we can make prediction with the trained neural network
print(model.predict(training_data).round())

# loss function:
# loss: 1.6401e-05
# the loss value is very very close to zero which means that the model is making good prediction
# In the first iteration, loss: 0.2485 (which means the model is 24.85% accurate)
# While training, the loss value keeps decreasing

# activation='sigmoid', epochs=5000
# loss: 1.2420e-05
# it means the loss function is very close to zero, there is a very very small error term
