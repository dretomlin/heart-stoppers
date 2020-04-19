#!/usr/bin/env python
# coding: utf-8

# Script for decision tree classification of heart disease data

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score, validation_curve
from sklearn import metrics
import os


# In[2]:


col_names = ['age', 'sex', 'chest-pain', 'restbps', 'cholesterol', 'fasting-bs', 'rest-ecg', 'thalach', 'exang', 'oldpeak', 'slope', 'colored-v', 'thal', 'num']
feature_cols = ['age', 'sex', 'chest-pain', 'restbps', 'cholesterol', 'fasting-bs', 'rest-ecg', 'thalach', 'exang', 'oldpeak', 'slope', 'colored-v', 'thal']


# load dataset

# In[3]:


heart = pd.read_csv("processed.cleveland.data", header=None, names=col_names)


# In[4]:


X = heart[feature_cols] # Features
y = heart['num'] # Target variable


# In[5]:


def data_clean(x_feat, y_label):
# Get names of indexes for which column Age has value 30
    for feat in feature_cols:
        indexNames = x_feat[x_feat[feat] == '?' ].index
        # Delete these row indexes from dataFrame
        x_feat.drop(indexNames , inplace=True)
        y_label.drop(indexNames , inplace=True)  
        
    X = x_feat.to_numpy()
    y = y_label.to_numpy()
    return(X, y)

X, y = data_clean(X, y)
X


# In[6]:


def initial_cv_score(clf_type, X_feat, y_label, num):
    cv_scores = cross_val_score(clf_type, X_feat, y_label, cv = num)
    sns.distplot(cv_scores)
    plt.title('Average score: {}'.format(np.mean(cv_scores)))
    
    directory = 'graph_pictures'
    
    if not os.path.exists('graph_pictures'):
        os.makedirs('graph_pictures')
    
    directory += '/dtree_init_cv_score.png'
    plt.savefig(directory)


# In[7]:


clf = DecisionTreeClassifier()


# In[8]:


initial_cv_score(clf, X, y, num=5)


# In[9]:


#Prints graph of validation curve for minimum number of leaves
def leaf_validation_plot(clf_type, X_feat, y_label, num):
    #Plot validation curve with leafs in nodes, DO with max depth
    # Create range of values for parameter
    param_range = np.arange(1, 50)

    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(clf_type, 
                                                 X_feat, 
                                                 y_label, 
                                                 param_name="min_samples_leaf", 
                                                 param_range=param_range,
                                                 cv=num, 
                                                 scoring="accuracy", 
                                                 n_jobs=-1)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots()
    
    # Plot mean accuracy scores for training and test sets
    ax.plot(param_range, train_mean, label="Training score", color="black")
    ax.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

    # Plot accurancy bands for training and test sets
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    ax.set_title("Validation Curve With Decision")
    ax.set_xlabel("Number Of Trees")
    ax.set_ylabel("Accuracy Score")
    #ax.tight_layout()
    ax.legend(loc="best")
    
    directory = 'graph_pictures'
    
    if not os.path.exists('graph_pictures'):
        os.makedirs('graph_pictures')
    
    directory += '/dtree_leaf_validate_scores.png'
    fig.savefig(directory)
    
    plt.show()


# In[10]:


leaf_validation_plot(clf, X, y, num=5)


# In[11]:


#Creating simple testing split to be used in model_fitter function
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 1)


# In[12]:


#GridSearchCV algorithmn to help combine an estimator with a grid search preamble to tune hyper-parameters
def gridSearch(clf, X_feat, y_label):
        
    # make an array of depths to choose from, say 1 to 20
    depths = np.arange(1, 21)
    num_leafs = np.arange(1,50)
    
    #GridSearch
    param_grid = [{'max_depth':depths, 'min_samples_leaf':num_leafs}]
    gs = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
    gs = gs.fit(X, y)
    print(gs.best_score_)
    print(gs.best_params_)
    #print(gs.best_estimator_)
    
    my_model = gs.best_estimator_    
    return(my_model)


# In[13]:


#Returns predictions and probabilities from model
def model_fitter(model):
    model.fit(X_train, y_train)
    y_predictor = model.predict(X_test)
    
    # predict probabilities
    dtree_probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    dtree_probs = dtree_probs[:, 1]
    return(y_predictor, dtree_probs)


# In[14]:


model = gridSearch(clf, X, y)


# In[15]:


#Get different types of probabilities from model
y_predictor, y_probs = model_fitter(model)


# In[16]:


#Working on function for evalution methods
def evals(y_test_truth, y_predictions):
    acc=metrics.accuracy_score(y_test_truth, y_predictions)
    mse=metrics.mean_squared_error(y_test_truth, y_predictions)
    c_report=metrics.classification_report(y_test_truth, y_predictions)
    
    print("Accuracy: ", acc)
    print("Mean-Squared Erro: ", mse)
    print("Classification report: ", c_report)
    


# In[17]:


evals(y_test,y_predictor)


# In[18]:


def make_confusion_matrix(y_test, y_predictor):
    cm=metrics.confusion_matrix(y_test, y_predictor)
    index = ['0-NP','1-P','2-P','3-P','4-P']  
    columns = ['0-NP','1-P','2-P','3-P','4-P']
    cm_df = pd.DataFrame(cm, columns, index)
    plt.figure(figsize=(10,6))  
    sns.heatmap(cm_df, annot=True)
    
    directory = 'graph_pictures'
    
    if not os.path.exists('graph_pictures'):
        os.makedirs('graph_pictures')
    
    directory += '/dtree_confusion_matrix.png'
    plt.savefig(directory)


# In[19]:


make_confusion_matrix(y_test, y_predictor)


# In[20]:


#Old Kfold method
'''
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

# Split dataset into training set and test set
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("MSE:",metrics.mean_squared_error(y_test, y_pred))
'''

