#Script for decision tree classification of heart disease data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer

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

# Convert dfs to numpy arrays
X = np.array(X)
y = np.array(y)

# Specify k folds
k = 5
skf = StratifiedKFold(n_splits=k)

# Begin training process
cur_fold = 1
n_classes = len(set(y))

accuracies_sum = 0
confusion_matrices_sum = 0
precisions_sum = 0
f1s_sum = 0
aucs_sum = 0
fprs_sum = 0
tprs_sum = 0
thresholds_sum = 0
y_pred_agg = []
y_test_agg = []
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

    y_pred_agg += list(y_pred)
    y_test_agg += list(y_test)

    # Compute metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracies_sum += accuracy

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    confusion_matrices_sum += confusion_matrix

    precision = metrics.precision_score(y_test, y_pred, average='micro')
    precisions_sum += precision

    f1 = metrics.f1_score(y_test, y_pred, average='micro')
    f1s_sum += f1

    y_test = [[num] for num in y_test]
    y_pred = [[num] for num in y_pred]
    y_test = MultiLabelBinarizer(classes=list(set(y))).fit_transform(y_test)
    y_pred = MultiLabelBinarizer(classes=list(set(y))).fit_transform(y_pred)
    
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    
    auc = metrics.roc_auc_score(y_test, y_pred, average='micro', multi_class='ovr')
    aucs_sum += auc

    # Compute micro-average ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
    fprs_sum += fpr
    tprs_sum += tpr

    cur_fold += 1


# Print average confusion matrix
print("Average confusion matrix:")
print("{}".format(confusion_matrices_sum/k))

# Print average f1
print("Average f1: {}".format(f1s_sum/k))

# Print average auc
print("Average auc: {}".format(aucs_sum/k))

# Plot average roc curve
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

plot_roc_curve(fprs_sum/k, tprs_sum/k)

#Working on function for evalution methods
def evals(y_test_truth, y_predictions):
    acc=metrics.accuracy_score(y_test_truth, y_predictions)
    mse=metrics.mean_squared_error(y_test_truth, y_predictions)
    c_report=metrics.classification_report(y_test_truth, y_predictions)

    print("Accuracy: ", acc)
    print("Mean-Squared Erro: ", mse)
    print("Classification report: ", c_report)

evals(y_test_agg, y_pred_agg)
