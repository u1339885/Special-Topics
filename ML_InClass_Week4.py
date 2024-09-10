#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:44:58 2024

@author: marcwhiting
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset
file_path = '/Users/marcwhiting/Library/Mobile Documents/com~apple~CloudDocs/Fall 2024/Chenglu/Week 4/Week4_Dataset (1).csv'
df = pd.read_csv(file_path)

# Define the predictors (all variables except grade_letter)
X = df[['score_quiz1', 'score_quiz2', 'score_quiz3', 'score_midterm', 'score_final', 'time_video', 
        'time_material', 'time_discussion', 'message_qna', 'message_discussion', 'message_twitter', 'assignment']]

# Convert grade_letter to numerical labels
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



# Convert the feature names to a list
feature_names = X.columns.tolist()

# Convert class names to a list
class_names = label_encoder.classes_.tolist()

# Plot the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(decision_tree, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()






