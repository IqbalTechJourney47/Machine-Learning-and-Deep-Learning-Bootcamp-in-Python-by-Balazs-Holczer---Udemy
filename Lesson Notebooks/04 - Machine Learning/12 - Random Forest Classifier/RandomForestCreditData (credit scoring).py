import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

credit_data = pd.read_csv('credit_data.csv')

features = credit_data[['income', 'age', 'loan']]
targets = credit_data.default

# machine learning handle arrays not data-frames
X = np.array(features).reshape(-1, 3)
# reshape(-1, 3) , 3 columns, -1 means python is going to figure out the number of rows which are 2000 in this case
# reshape(x, y)

y = np.array(targets)

model = RandomForestClassifier()
predicted = cross_validate(model, X, y, cv=10)

print(np.mean(predicted['test_score']))
