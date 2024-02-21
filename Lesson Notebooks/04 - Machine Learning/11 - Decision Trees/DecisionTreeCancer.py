import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import cross_validate

cancer_data = datasets.load_breast_cancer()

print(cancer_data.data)

# 30 features in the dataset

print(cancer_data.target)
# 1-D array with 0 and 1

features = cancer_data.data
labels = cancer_data.target

print(features.shape)

print(labels.shape)

feature_train, feature_test, target_train, target_test = train_test_split(features, labels, test_size=0.3, random_state=101)

model = DecisionTreeClassifier(criterion='entropy' , max_depth=5)

predicted = cross_validate(model, features, labels, cv=10)
print(np.mean(predicted['test_score']))
