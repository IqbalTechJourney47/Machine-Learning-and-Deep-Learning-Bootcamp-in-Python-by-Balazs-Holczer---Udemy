import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()

print(digits.images)
# 8 columns and 8 rows because we are dealing with 8*8 images

print(digits.images.shape)
# (1797, 8, 8)
# every single item in the dataset has 8 rows and 8 columns (single image)

print(digits.target)
# 8*8 (2D array)
# 8*8 - 0
# 8*8 - 1
# 8*8 - 2
# assign a single digit to each image

# 1-D array with 1797 items. This is the size of the dataset

images_and_labels = list(zip(digits.images, digits.target))
print(images_and_labels)

# it is a supervised learning algorithm
# it has the labels and the images accordingly

# plot where images and labels are concerned
# [:6] - first 6 items
for index, (image, label) in enumerate(images_and_labels[:6]):

    # creating a subplot with 2 rows and 3 columns
    # we are going to plot the image and the label accordingly
    plt.subplot(2, 3, index+1)

    # imshow function shows the image as far as the gray scale image is concerned
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

    # we are going to set the title as the target and we are going to show the label
    plt.title('Target: %i' %label)

plt.show()

# we would like to reshape it because we would like to flatten the data

# We have to transform the images into numerical values
# and we can do it with the help of pixel intensities

# to apply a classifier on this data, we need to flatten the image: instead of a 8x8 matrix
# 8x8 matrix - 2D array
# we have to use a 1-D array with 64 items

# -1 means python is going to calculate the parameter for us
# In this case, we are after the 1-D representation of that given matrix
data = digits.images.reshape((len(digits.images), -1))

# -1 means python is going to calculate the parameter for us
# In this case, we are after the 1-D representation of that given matrix
# it means if we print out the data
# print(data)
# it is a 2D array with sub-arrays and every single sub-array contains 64 pixel intensities

# This is going to represent the 1st image in first sub-array, 2nd image in 2nd sub-array and so on

# Every single row in going to contain 64 items because of (8x8) image size

print(data)

classifier = svm.SVC(gamma=0.001)

# 75% of the original dataset is for training
# (len(digits.images) is going to represent whole dataset with 1797 items
train_test_split = int(len(digits.images) * 0.75)
# here [:train_test_split], we started with zero and up to the train_test_split, so up to the 75% of the dataset
classifier.fit(data[:train_test_split], digits.target[:train_test_split])

# Now predict the value of the digit on the 25%
# expected - we know for certain because it is a supervised learning algorithm
# This is why we have the target labels as digits target starting from [train_test_split:] to the end of the dataset
# here [train_test_split:], we started with 75% and we up to that 100% of the dataset
expected = digits.target[train_test_split:]
# here we make predicts with the help of support vector classifier on the test data set (data[train_test_split:])
# expected = digits.target[train_test_split:] here we have values we know from test dataset
# and we have the predictions for the task
predicted = classifier.predict(data[train_test_split:])

print('Confusion matrix:\n%s' %metrics.confusion_matrix(expected, predicted))
print(accuracy_score(expected, predicted))

# Let's test on the last few images
# digits.images[-1] - last image in the dataset
# classifier.predict(data[-1].reshape(1, -1)) - last image in the dataset
# digits.images[-2] - item/image before the last image
# classifier.predict(data[-2].reshape(1, -1)) - item/image before the last image
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation='nearest')
print('Prediction for test image: ', classifier.predict(data[-2].reshape(1, -1)))

plt.show()
