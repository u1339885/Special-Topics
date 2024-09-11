#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:25:26 2024

@author: marcwhiting
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset
file_path = 'FilePath.csv'
df = pd.read_csv(file_path)

# Define the predictors (independent variables: score_quiz1 and time_video)
X = df[['score_quiz1', 'time_video']]

# Convert grade_letter to numerical labels (dependent variable)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['grade_letter'])

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Predict the grade letters on the test data
y_pred = decision_tree.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Convert the class names to a list (to avoid the error)
class_names = label_encoder.classes_.tolist()

# Plot the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(decision_tree, feature_names=['score_quiz1', 'time_video'], class_names=class_names, filled=True)
plt.show()

