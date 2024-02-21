# Logistic Regression - credit scoring

# Multiple logistic problem

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

credit_data = pd.read_csv('credit_data.csv')

# print(credit_data.head())
# LTI = loan to income ratio

# print(credit_data.describe())

print(credit_data.corr())
# In corr, you can check what is the relationship between these features

features = credit_data[['income', 'age', 'loan']]
target = credit_data.default

# 30% of the data-set is for testing and 70% of the data-set is for training
# This means that we are going to use 70% of the dataset in order to find the b values
# b0, b1, b2 and b3
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(feature_train, target_train)
# finding b0, b1, b2, b3

print(model.fit.coef_)
print(model.fit.intercept_)

predictions = model.fit.predict(feature_test)
# as far as the features are concerned and the test dataset is concerned
# we are going to feed the features of the test dataset to the model
# and we make predictions
# and then we can calculate the accuracy

print(confusion_matrix(target_test, predictions))

# confusion matrix
# 503 items are classified correctly as far as 0 is concerned, 0 - default
# 67 items are classified correctly as far as 1 is concerned, 1 - default

print(accuracy_score(target_test, predictions))

