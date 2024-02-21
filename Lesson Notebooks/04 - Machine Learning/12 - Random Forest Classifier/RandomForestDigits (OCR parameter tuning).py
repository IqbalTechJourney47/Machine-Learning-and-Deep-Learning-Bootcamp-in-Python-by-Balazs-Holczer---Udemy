# OCR = Optical Character Recognition

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

digit_data = datasets.load_digits()

# As there are 8x8 pixels, this is why there are going to be 64 features

print(digit_data)

# we would like to reshape it because we would like to flatten the data

# We have to transform the images into numerical values
# and we can do it with the help of pixel intensities

# to apply a classifier on this data, we need to flatten the image: instead of a 8x8 matrix
# 8x8 matrix - 2D array
# we have to use a 1-D array with 64 items

# -1 means python is going to calculate the parameter for us
# In this case, we are after the 1-D representation of that given matrix
image_features = digit_data.images.reshape((len(digit_data.images), -1))

# -1 means python is going to calculate the parameter for us
# In this case, we are after the 1-D representation of that given matrix
# it means if we print out the data
# print(image_features)
# it is a 2D array with sub-arrays and every single sub-array contains 64 pixel intensities

# This is going to represent the 1st image in first sub-array, 2nd image in 2nd sub-array and so on

# Every single row in going to contain 64 items because of (8x8) image size

image_targets = digit_data.target
# image_targets - 1-D array with 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 because these are the possible values

print(image_features)

print(image_targets)

print(image_targets.shape)
# (1797,)

print(image_features.shape)
# (1797, 64)
# 1797 rows and every single row has 64 columns

random_forest_model = RandomForestClassifier(n_jobs=-1, max_features='sqrt')
# max_features are taken by trees randomly like for each iteration in random forest classifier
# n_jobs, the number of jobs to run in parallel
# -1 means using all processors cores
# we would like to use as many processors as possible with parallel computing
# n_jobs, the number of jobs has something to do with parallel computing

feature_train, feature_test, target_train, target_test = train_test_split(image_features, image_targets, test_size=0.3, random_state=101)

param_grid = {
    'n_estimators': [10, 100, 500, 1000],
    'max_depth': [1, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 10, 15, 30, 50]
}
# n_estimators, the number of trees in the forest
# max_depth, the maximum depth of a given tree
# min_samples_leaf, the minimum number of samples required to be at a leaf node

grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=10)
grid_search.fit(feature_train, target_train)
print(grid_search.best_params_)

optimal_estimators = grid_search.best_params_.get('n_estimators')
optimal_depth = grid_search.best_params_.get('max_depth')
optimal_leaf = grid_search.best_params_.get('min_samples_leaf')

print('Optimal n_estimators: %s' %optimal_estimators)
print('Optimal optimal_depth %s' %optimal_depth)
print('Optimal optimal_leaf %s' %optimal_leaf)

grid_predictions = grid_search.predict(feature_test)

print(confusion_matrix(target_test, grid_predictions))
print(accuracy_score(target_test, grid_predictions))
