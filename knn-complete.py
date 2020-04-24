#!/usr/bin/env python

import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd 
from statistics import mean
from sklearn import metrics
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import os

col_names = ['age', 'sex', 'chest-pain', 'restbps', 'cholesterol', 'fasting-bs', 'rest-ecg', 'thalach', 'exang', 'oldpeak', 'slope', 'colored-v', 'thal', 'num']
feature_cols = ['age', 'sex', 'chest-pain', 'restbps', 'cholesterol', 'fasting-bs', 'rest-ecg', 'thalach', 'exang', 'oldpeak', 'slope', 'colored-v', 'thal']

# Load dataset
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

scaled_features = preprocessing.StandardScaler().fit_transform(X.values)
X = scaled_features
# Convert dfs to numpy arrays
y = np.array(y)

# Specify k folds
k = 5
skf = StratifiedKFold(n_splits=k)

def initial_cv_score(clf_type, X_feat, y_label, num):
    cv_scores = cross_val_score(clf_type, X_feat, y_label, cv = num)
    sns.distplot(cv_scores)
    plt.title('Average score: {}'.format(np.mean(cv_scores)))
    
    directory = 'graph_pictures'
    
    if not os.path.exists('graph_pictures'):
        os.makedirs('graph_pictures')
    
    directory += '/knn_init_cv_score.png'
    plt.savefig(directory)
    plt.close()


knn = KNeighborsClassifier()
initial_cv_score(knn, X, y, num=5)

def best_k(X, y):
    # try K=1 through K=25 and record testing accuracy
    k_range = range(1, 26)

    # We can create Python dictionary using [] or dict()
    scores = []

    # We use a loop through the range 1 to 26
    # We append the scores in the dictionary
    
    for train_index, test_index in skf.split(X, y):
        # Split data
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))

   # plot the relationship between K and testing accuracy
    # plt.plot(x_axis, y_axis)
    plt.plot(k_range, scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    
    directory = 'graph_pictures'

    if not os.path.exists('graph_pictures'):
        os.makedirs('graph_pictures')

    directory += '/knn_best_k.png'
    plt.savefig(directory)
    plt.close()

    return(X_train, X_test, y_train, y_test)
    
X_train, X_test, y_train, y_test = best_k(X,y)

def knn_grid(X,y):
    #create new a knn model
    knn2 = KNeighborsClassifier()#create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(1, 10)}#use gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(knn2, param_grid, cv=5)#fit model to data
    knn_gscv.fit(X, y)
    #check top performing n_neighbors value
    print(knn_gscv.best_params_)
    #check mean score for the top performing value of n_neighbors
    print(knn_gscv.best_score_)
    print(knn_gscv.best_estimator_)
    return(knn_gscv.best_estimator_)

model = knn_grid(X,y)

#Returns predictions and probabilities from model
def model_fitter(model):
    model.fit(X_train, y_train)
    y_predictor = model.predict(X_test)
    
    # predict probabilities
    dtree_probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    dtree_probs = dtree_probs[:, 1]
    return(y_predictor, dtree_probs)

#Get different types of probabilities from model
y_predictor, y_probs = model_fitter(model)

#Working on function for evalution methods
def evals(y_test_truth, y_predictions):
    acc=metrics.accuracy_score(y_test_truth, y_predictions)
    mse=metrics.mean_squared_error(y_test_truth, y_predictions)
    c_report=metrics.classification_report(y_test_truth, y_predictions)

    print("Accuracy: ", acc)
    print("Mean-Squared Erro: ", mse)
    print("Classification report: ", c_report)

evals(y_test, y_predictor)

#essentially redoing the whole program, but with binarized y labels for per class 
#roc curve creation
def multiclass_split(X, y, model):
    y = preprocessing.label_binarize(y, classes=[0,1,2,3,4])
    n_classes=5
   
    #Create Kfold
    kf = KFold(5, True, 1) # Define the split - into 2 folds 
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    y_train_int = y_train.astype('int')
    y_test_int = y_test.astype('int')
    model2 = OneVsRestClassifier(model)
    model2.fit(X_train, y_train)
    
    #Create prediction
    probs = model2.predict_proba(X_test)
    probs

    #Plot ROC Curve per class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    roc_summer = []

    for i in range(5):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], probs[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        print(roc_auc[i])
        roc_summer.append(roc_auc[i])
    
    print("Average ROC-AUC: ", sum(roc_summer) / len(roc_summer))

    colors = ['blue', 'red', 'green', 'cyan', 'magenta']
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=3)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    
    directory = 'graph_pictures'

    if not os.path.exists('graph_pictures'):
        os.makedirs('graph_pictures')

    directory += '/knn_roc_curve.png'
    plt.savefig(directory)
    
    plt.show()
multiclass_split(X,y,model)
