# 006 Support vector machine example II - iris dataset

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

iris_data = datasets.load_iris()

print(iris_data.data)

# sepal length, sepal width, petal length, petal width

print(iris_data.data.shape)
# (150, 4)

print(iris_data.target)

print(iris_data.target.shape)
# (150,)

features = iris_data.data
target = iris_data.target

feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=101)

model = svm.SVC()
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
