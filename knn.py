#Script for decision tree classification of heart disease data

import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

col_names = ['age', 'sex', 'chest-pain', 'restbps', 'cholesterol', 'fasting-bs', 'rest-ecg', 'thalach', 'exang', 'oldpeak', 'slope', 'colored-v', 'thal', 'num']
feature_cols = ['age', 'sex', 'chest-pain', 'restbps', 'cholesterol', 'fasting-bs', 'rest-ecg', 'thalach', 'exang', 'oldpeak', 'slope', 'colored-v', 'thal']

#load dataset
heart = pd.read_csv("processed.cleveland.data", header=None, names=col_names)

X = heart[feature_cols].copy() # Features
y = heart.num.copy() # Target variable

# Clean data
for feature in feature_cols:
    if X[feature].dtype == np.float64:
        continue
    indexNames = X[X[feature] == '?' ].index
    # Delete these row indexes from dataFrame
    X.drop(indexNames , inplace=True)
    y.drop(indexNames , inplace=True)
    X[feature] = X[feature].astype(np.float64)



# Convert dfs to numpy arrays
X = np.array(X)
y = np.array(y)

# Specify k folds
k = 5
skf = StratifiedKFold(n_splits=k)

# Begin training process
cur_fold = 1
print("Fold: Accuracy")
accuracies = 0
for train_index, test_index in skf.split(X, y):
    # Split data
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    # Train classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Print accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracies += accuracy
    print("   {}: {}".format(cur_fold, accuracy))
    cur_fold += 1

# Print average accuracy
print("\nAverage accuracy: {}".format(accuracies/k))
