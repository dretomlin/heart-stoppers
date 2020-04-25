import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection  import KFold, cross_val_score
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#all Ploting stuff based on that from dTree file, modified to work for this program
# In[2]
col_names = ['age', 'sex', 'chest-pain', 'restbps', 'cholesterol', 'fasting-bs', 'rest-ecg', 'thalach', 'exang', 'oldpeak', 'slope', 'colored-v', 'thal', 'num']
feature_cols = ['age', 'sex', 'chest-pain', 'restbps', 'cholesterol', 'fasting-bs', 'rest-ecg', 'thalach', 'exang', 'oldpeak', 'slope', 'colored-v', 'thal']
# load dataset
# In[3]:
heart = pd.read_csv("processed.cleveland.data", header=None, names=col_names)
# In[4]:
X = heart[feature_cols] # Features
y = heart['num'] # Target variable

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

cnb=SVC(gamma='auto')

cnb.fit(X_train, y_train)

y_pred=cnb.predict(X_test)

'''
metrics Section
'''

#accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("accuracy:")
print(accuracy)


#classification_report
print("classification_report:")
print(metrics.classification_report(y_test, y_pred))

#Kfolds and cv_score
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
print("cross_val_score:")
cv_scores=cross_val_score(cnb, X, y, cv=k_fold, n_jobs=1)
print(cv_scores)
sns.distplot(cv_scores)
plt.title('Average score: {}'.format(np.mean(cv_scores)))
    
directory = 'graph_pictures'
    
if not os.path.exists('graph_pictures'):
    os.makedirs('graph_pictures')
    
directory += '/skn_init_cv_score.png'
plt.savefig(directory)

#precision
precision = metrics.precision_score(y_test, y_pred, average='micro')
print("precision:")
print(precision)

#confusion matrix.
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
    
    directory += '/skn_confusion_matrix.png'
    plt.savefig(directory)

make_confusion_matrix(y_test, y_pred)
#MSE
mse=metrics.mean_squared_error(y_test, y_pred)
print("MSE:")
print(mse)

#recall score:
rscore=metrics.recall_score(y_test.ravel(), y_pred.ravel(), average=None)
print("recall score")
print(rscore)

#f1 score:
f1score=metrics.f1_score(y_test.ravel(), y_pred.ravel(), average=None)
print("F1 Score:")
print(f1score)
#learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    directory = 'graph_pictures'
    
    if not os.path.exists('graph_pictures'):
        os.makedirs('graph_pictures')
    
    directory += '/skn_learning_curve.png'
    plt.savefig(directory)

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
plot_learning_curve(cnb, title, X, y, (0.4, 1.01), cv=cv, n_jobs=4)
