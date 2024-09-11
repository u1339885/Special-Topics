#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:16:17 2024

@author: marcwhiting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load the dataset
file_path = 'FilePath.CSV'
df = pd.read_csv(file_path)

# Define the independent variables and the dependent variable
X = df[['message_discussion', 'message_twitter', 'time_discussion']]
y = df['grade_score']

# Standardize the independent variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add a constant to the independent variables (for the intercept)
X_scaled_with_constant = sm.add_constant(X_scaled)

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store coefficients and p-values
coefficients = []
p_values = []

# For each fold, fit the model and store the results
for train_index, test_index in kf.split(X_scaled_with_constant):
    X_train, X_test = X_scaled_with_constant[train_index], X_scaled_with_constant[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Fit the OLS model
    model = sm.OLS(y_train, X_train).fit()
    
    # Append the coefficients and p-values from each fold
    coefficients.append(model.params[1:])  
    p_values.append(model.pvalues[1:])     

# Convert the results to numpy arrays for easier manipulation
coefficients = np.array(coefficients)
p_values = np.array(p_values)

# Calculate the mean coefficients and p-values across the folds
mean_coefficients = coefficients.mean(axis=0)
mean_p_values = p_values.mean(axis=0)

# Display the standardized coefficients and their corresponding p-values
for i, column in enumerate(X.columns):
    print(f"{column}: Coefficient = {mean_coefficients[i]}, P-value = {mean_p_values[i]}")
