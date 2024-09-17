#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:27:13 2024

@author: marcwhiting
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, r2_score, make_scorer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'FilePath.csv'
data = pd.read_csv(file_path)

# Select the features and target for the model
features = ['disability', 'num_of_prev_attempts', 'studied_credits']
target = 'final_result'

# Convert categorical variables into numeric
# Encode 'disability' (assuming it's categorical)
data['disability'] = data['disability'].apply(lambda x: 1 if x == 'Y' else 0)

# Encode 'final_result' to binary (1 for 'Withdrawn', 0 otherwise)
data['final_result'] = data['final_result'].apply(lambda x: 1 if x == 'Withdrawn' else 0)

# Select relevant data
X = data[features]
y = data[target]

# Initialize the gradient boosting classifier
model = GradientBoostingClassifier(random_state=42)

# Perform 5-fold cross-validation for accuracy
accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
mean_accuracy = accuracy_scores.mean()

# Perform 5-fold cross-validation for R²
r2_scorer = make_scorer(r2_score)
r2_scores = cross_val_score(model, X, y, cv=5, scoring=r2_scorer)
mean_r2 = r2_scores.mean()

# Train the model on the full dataset to get feature importances
model.fit(X, y)
feature_importance = model.feature_importances_

# Display the results
print(f"Coefficient of Determination (R²): {mean_r2:.4f}")
print(f"Accuracy: {mean_accuracy:.4f}")
print(f"Feature Importances:")
for feature, importance in zip(features, feature_importance):
    print(f"  {feature}: {importance:.4f}")
