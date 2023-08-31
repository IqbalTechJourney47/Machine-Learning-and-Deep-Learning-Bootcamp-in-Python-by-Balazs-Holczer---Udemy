from PIL import Image
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

directory = 'training_set/'

# [
#     [0, 10, 25, ..., 100, 150, 200, ...],  # Pixel intensities for the first image
#     [255, 245, 230, ..., 155, 105, 50, ...],  # Pixel intensities for the second image
# ]

# You open the image using the PIL (Pillow) library's Image.open method.
# .convert('1') converts the image to grayscale using a single channel, meaning that each pixel value represents the intensity of the gray color.


# pixel_intensities.append(list(image.getdata())):

# You're extracting the pixel intensities from the resized grayscale image using the .getdata() method. This method returns a sequence of pixel values as a 1D tuple.

# list(image.getdata()) converts the tuple of pixel values into a list and appends it to the pixel_intensities list.

# Define a common image size
common_image_size = (32, 32)  # Adjust as needed

pixel_intensities = []  # 1D array, List to store resized pixel intensities

# one-hot encoding: happy (1,0) and sad (0,1)
labels = []

for filename in os.listdir(directory):
    # print(filename)
    image = Image.open(directory + filename).convert('1')
    # image = Image.open(os.path.join(directory, filename)).convert('L')  # Convert to grayscale
    image = image.resize(common_image_size)  # Resize the image to a common size
    # print(image)

    pixel_intensities.append(list(image.getdata()))
    # print(image.getdata())

    # pixel_intensities.append(np.array(image).flatten())  # Convert to numpy array and flatten
    # print(np.array(image).flatten())

    # print(pixel_intensities) # Pixel intensities for the first image, for the second image and so on.

    # print(list(image.getdata())) # converts the tuple of pixel values into a list and appends it to the pixel_intensities list

    if filename[0:5] == 'happy':  # [0:5] - first 5 characters
        labels.append([1, 0])
    elif filename[0:3] == 'sad':
        labels.append([0, 1])

    # if filename.startswith('happy'):
    #    labels.append([1, 0])
    # elif filename.startswith('sad'):
    #    labels.append([0, 1])

pixel_intensities = np.array(pixel_intensities)
# print(pixel_intensities)   # it returns 1D array containing sub-arrays
# where every single sub-array or every single row in the dataset is going to represent
# a given image with pixel_intensities
# 32*32 images
# 1024 items in a given sub-array=

# print(pixel_intensities.shape) # it returns (30, 1024) ) (6, 640000) 6 is the size of the dataset

labels = np.array(labels)
# print(labels.shape) # it returns (30, 2) 30 samples in the dataset, 2 columns/value happy(1,0) and sad(0,1)
# print(labels) # it retuens a 1D array with sub-arrays
# every single sub-array is going to represent the label associated with a given face

labels = np.array(labels)

# Now pixel_intensities will be a 2D array with consistent shape, suitable for training


# In this modified code, images are resized to a common size (common_image_size) before converting them to numpy arrays.
# The images are converted to grayscale using the 'L' mode for simplicity.
# The pixel intensities are flattened into a 1D array using flatten(), and then the processed data is used to create
# the numpy arrays pixel_intensities and labels.

# apply min-max normalization (here just /255)
# transformation of values within the range 0 and 1

pixel_intensities = pixel_intensities / 255.0

# print(pixel_intensities)

# Create the model (deep neural networks)
model = Sequential()
model.add(Dense(1024, input_dim=1024, activation='relu')) # input layer
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax')) # output layer

learning_rate = 0.005
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy',
             optimizer=optimizer,
             metrics=['accuracy'])

model.fit(pixel_intensities, labels, epochs=1000, batch_size=20, verbose=2) # pixel_intensities, as far as the features are concerned

# handle the test dataset ( images)
print("Testing the neural network....")

test_pixel_intensities = []

test_image1 = Image.open('test_set/happy_test1.png').convert(
    '1')  # we have to transform the image into 1D numpy array as far
# as the pixel intensities are concerned
test_pixel_intensities.append(list(test_image1.getdata()))  # (list(test_image1.getdata())
# we get the pixel intensities, we transform it into a list
# and then we all did as the first item in 1D array
# test_pixel_intensities = []

# we have to normalize the data again

test_pixel_intensities = np.array(test_pixel_intensities) / 255

print(model.predict(test_pixel_intensities).round())
