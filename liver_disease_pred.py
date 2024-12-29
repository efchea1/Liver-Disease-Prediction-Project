#!/usr/bin/env python
# coding: utf-8

# In[7]:


# load needed libraries
import pandas as pd # for working with structured data in the form of dataframes, enabling data manipulation, analysis, and preprocessing.
import numpy as np # for numerical computations, such as working with arrays and performing mathematical operations.

# split the training data
from sklearn.model_selection import train_test_split # function from scikit-learn to divide the dataset into training and testing subsets for model evaluation.
from sklearn.preprocessing import StandardScaler # normalize features by removing the mean and scaling them to unit variance, which ensures consistent feature ranges for better model performance.

# import needed ML algorithms
from sklearn.ensemble import RandomForestClassifier # a machine learning model based on an ensemble of decision trees, commonly used for classification tasks.
from sklearn.svm import SVC # a machine learning model that uses hyperplanes for classification, suitable for both linear and non-linear problems.
from sklearn.neural_network import MLPClassifier # a neural network-based classifier that uses the multilayer perceptron algorithm for predictive modeling.

# for model evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # classification_report: Summarizes precision, recall, F1-score, and support for each class.
# confusion_matrix: Shows a matrix comparing predicted vs. actual classes.
# accuracy_score: Measures the overall accuracy of the model.

# Load the dataset
data = pd.read_csv(r"C:\Users\emman\OneDrive\Desktop\Data Science Projects\Liver Disease Prediction\Liver_disease_data.csv")

# Exploratory Data Analysis
print(data.info())
print(data.describe())

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Features and target
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
print("Random Forest Performance:")
print(classification_report(y_test, rf_predictions))

# Model 2: Support Vector Machine
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_predictions = svm_model.predict(X_test_scaled)
print("SVM Performance:")
print(classification_report(y_test, svm_predictions))

# Model 3: Neural Network (MLP) with increased iterations and early stopping
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, solver='adam', random_state=42, early_stopping=True, n_iter_no_change=20)
mlp_model.fit(X_train_scaled, y_train)
mlp_predictions = mlp_model.predict(X_test_scaled)
print("Neural Network Performance:")
print(classification_report(y_test, mlp_predictions))

# Compare model accuracy
rf_accuracy = accuracy_score(y_test, rf_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
mlp_accuracy = accuracy_score(y_test, mlp_predictions)

print("\nModel Comparison:")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"Neural Network Accuracy: {mlp_accuracy:.2f}")

# Save the best model
best_model = None
if rf_accuracy >= max(svm_accuracy, mlp_accuracy):
    best_model = rf_model
elif svm_accuracy >= max(rf_accuracy, mlp_accuracy):
    best_model = svm_model
else:
    best_model = mlp_model

print("Best model saved based on accuracy.")


# In[ ]:




