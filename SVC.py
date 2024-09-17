#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:39:42 2024

@author: marcwhiting
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, make_scorer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/marcwhiting/Library/Mobile Documents/com~apple~CloudDocs/Fall 2024/Chenglu/Assignments/Data Analysis 3/Week5 Dataset.csv'
data = pd.read_csv(file_path)

# Select the features and target for the SVM model
features = ['disability', 'num_of_prev_attempts', 'sum(sum_click)_page', 'sum(sum_click)_quiz']
target = 'final_result'

# Data preprocessing
# Encode 'disability' as binary (1 if 'Y', 0 otherwise)
data['disability'] = data['disability'].apply(lambda x: 1 if x == 'Y' else 0)

# Convert 'final_result' to binary (1 for 'Withdrawn', 0 otherwise)
data['final_result'] = data['final_result'].apply(lambda x: 1 if x == 'Withdrawn' else 0)

# Handle missing values by filling with 0
data[features] = data[features].fillna(0)

# Extract relevant data
X = data[features]
y = data[target]

# Scale the features for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize the SVM classifier with a linear kernel
svm_model = SVC(kernel='linear', probability=True, random_state=42)

# Cross-validate accuracy
accuracy_scores = cross_val_score(svm_model, X_scaled, y, cv=5, scoring='accuracy')
mean_accuracy = accuracy_scores.mean()

# Cross-validate AUC
auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
auc_scores = cross_val_score(svm_model, X_scaled, y, cv=5, scoring=auc_scorer)
mean_auc = auc_scores.mean()

# Train the model on the full training set to extract the coefficients (weights)
svm_model.fit(X_train, y_train)
weights = svm_model.coef_[0]

# Create a weight table
weight_table = pd.DataFrame({'Feature': features, 'Weight': weights})

# Get the predicted probabilities for ROC curve
y_pred_proba = svm_model.predict_proba(X_test)[:, 1]

# Print the cross-validated results
print(f"Cross-Validated Accuracy: {mean_accuracy:.4f}")
print(f"Cross-Validated AUC Value: {mean_auc:.4f}")
print("\nWeight Table:")
print(weight_table)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {mean_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
