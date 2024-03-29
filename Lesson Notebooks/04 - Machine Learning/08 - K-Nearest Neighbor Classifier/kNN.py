import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

x_blue = np.array([0.3, 0.5, 1, 1.4, 1.7, 2])
y_blue = np.array([1, 4.5, 2.3, 1.9, 8.9, 4.1])

x_red = np.array([3.3, 3.5, 4, 4.4, 5.7, 6])
y_red = np.array([7, 1.5, 6.3, 1.9, 2.9, 7.1])

X = np.array([[0.3, 1], [0.5, 4.5], [1, 2.3], [1.4, 1.9], [1.7, 8.9], [2, 4.1],
              [3.3, 7], [3.5, 1.5], [4, 6.3], [4.4, 1.9], [5.7, 2.9], [6, 7.1]])

y= np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# ro - round circles
plt.plot(x_blue, y_blue, 'ro', color='blue')
plt.plot(x_red, y_red, 'ro', color='red')

# given input, x=3, y=5
plt.plot(3, 5, 'ro', color='green', markersize=15)
# plt.plot(1, 5, 'ro', color='green', markersize=15)
# plt.plot(8, 5, 'ro', color='green', markersize=15)
# x-value = 3,      y-value = 5

plt.axis([-0.5, 10, -0.5, 10])
# horizontal axis ranges from -0.5-10
# vertical axis ranges from -0.5-10

# plt.show()

# graph
# if x is small, then color is blue
# if x is larger, then approximately the color is going to be red

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X, y)

predict = classifier.predict(np.array([[5, 5]]))
# predict = classifier.predict(np.array([[1, 5]]))
# predict = classifier.predict(np.array([[8, 5]]))
print(predict)

plt.show()
