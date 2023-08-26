import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

credit_data = pd.read_csv('credit_data.csv')

features = credit_data[['income', 'age', 'loan']]
target = credit_data.default

# print(features)

# machine learning handle arrays not data-frames
X = np.array(features).reshape(-1, 3)
# reshape(-1, 3) , 3 columns, -1 means python is going to figure out the number of rows which are 2000 in this case

y = np.array(target)

# print(X)
# print(y)

model = LogisticRegression()
predicted = cross_validate(model, X, y, cv=5)
# cv = cross validation
# predicted = cross_validate(model, X, y, cv=10)

print(predicted['test_score'])
# sklearn.model_selection.cross_validate
# test_score = The score array for test scores on each cv split

print(np.mean(predicted['test_score']))
# it is the accuracy of logistic regression with the help of cross validation
