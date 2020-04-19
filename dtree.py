#Script for decision tree classification of heart disease data

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names = ['age', 'sex', 'chest-pain', 'restbps', 'cholesterol', 'fasting-bs', 'rest-ecg', 'thalach', 'exang', 'oldpeak', 'slope', 'colored-v', 'thal', 'num']
feature_cols = ['age', 'sex', 'chest-pain', 'restbps', 'cholesterol', 'fasting-bs', 'rest-ecg', 'thalach', 'exang', 'oldpeak', 'slope', 'colored-v', 'thal']

#load dataset

heart = pd.read_csv("processed.cleveland.data", header=None, names=col_names)

X = heart[feature_cols] # Features
y = heart.num # Target variable

# Get names of indexes for which column Age has value 30
for feature in feature_cols:
    indexNames = X[X[feature] == '?' ].index
    # Delete these row indexes from dataFrame
    X.drop(indexNames , inplace=True)
    y.drop(indexNames , inplace=True)



# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

