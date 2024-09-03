# Training and testing of csv files

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, zero_one_loss
from PlotConfusionMatrix import plotMatrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


raw_data_filename = "data/expendData/total_extend.csv"
# raw_data_filename = "extracted_features.csv"
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_filename, header=None, skiprows=1, low_memory=False, encoding='latin1')

# Random sampling ratio, this can be used when the dataset is large, optional
raw_data = raw_data.sample(frac=0.1)

# Check the status of the labelled data
last_column_index = raw_data.shape[1] - 1

# Converting non-numeric data to numeric data
print("Transforming data...")
for column in raw_data.columns:
    if raw_data[column].dtype == 'object':
        raw_data[column], attacks = pd.factorize(raw_data[column], sort=True)
# raw_data[last_column_index], attacks = pd.factorize(raw_data[last_column_index], sort=True)

# Slicing the raw data to isolate features and labels
features = raw_data.iloc[:, :raw_data.shape[1] - 1]  # The iloc slices in pandas are fully position-based indexes
labels = raw_data.iloc[:, raw_data.shape[1] - 1:]
print("Initial number of columns in features:", features.shape[1])

unique, counts = np.unique(labels, return_counts=True)
label_distribution = dict(zip(unique, counts))

print("Label distribution:", label_distribution)
print("Number of columns in features after removing object type columns:", features.shape[1])

# Feature data standardisation, optional
features = preprocessing.scale(features)
features = pd.DataFrame(features)

# Converting multi-dimensional labels to one-dimensional arrays
labels = labels.values.ravel()

# Divide the data into training and test sets and print the dimensionality
df = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2, stratify=labels)

print("X_train,y_train:", X_train.shape, y_train.shape)
print("X_test,y_test:", X_test.shape, y_test.shape)
print("Training model...")

# decision tree
# clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_leaf=1, splitter="best")

# SVM
# clf = LinearSVC(C=0.01, max_iter=2000)  # kernel can choose ‘linear’, ‘rbf’, ‘poly’ etc. C is the regularisation parameter

# random forest
# clf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)

# knn
# clf = KNeighborsClassifier(n_neighbors=5)  # n_neighbors is the value of K

# Logistic Regression
# clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='auto')

# Simple Bayesian
clf = GaussianNB()

trained_model = clf.fit(X_train, y_train)
print("Score:", trained_model.score(X_train, y_train))

# predict
print("Predicting...")
y_pred = clf.predict(X_test)
print("Computing performance metrics...")

results = confusion_matrix(y_test, y_pred)
error = zero_one_loss(y_test, y_pred)
# Prediction accuracy based on confusion matrix
list_diag = np.diag(results)
list_raw_sum = np.sum(results, axis=1)
print("Predict accuracy of the decisionTree: ", np.mean(list_diag) / np.mean(list_raw_sum))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # ‘weighted’ for multi-categorisation problems
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Plotting the confusion matrix
plotMatrix(attacks, y_test, y_pred)