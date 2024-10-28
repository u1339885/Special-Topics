#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:47:23 2024

@author: marcwhiting
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/marcwhiting/Library/Mobile Documents/com~apple~CloudDocs/Fall 2024/Chenglu/oct_26/Weel10_Dataset 2.csv'
try:
    data = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Extract the abstracts column
documents = data['abstracts'].dropna().tolist()

# Preprocessing: Vectorize the text data
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
doc_term_matrix = vectorizer.fit_transform(documents)

# Function to fit LDA models and calculate log-likelihood
def fit_lda(num_topics, doc_term_matrix):
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)
    log_likelihood = lda.score(doc_term_matrix)
    return lda, log_likelihood

# Fit LDA models with different number of topics and store log-likelihood
results = {}
for num_topics in [2, 3, 4, 5]:
    lda_model, log_likelihood = fit_lda(num_topics, doc_term_matrix)
    results[num_topics] = (lda_model, log_likelihood)

print("\nLog-Likelihood values:")
log_likelihood_values = {k: v[1] for k, v in results.items()}
for num_topics, log_likelihood in log_likelihood_values.items():
    print(f'{num_topics}-Topic model: {log_likelihood:.2f}')

# Choose the model with the highest log-likelihood
best_num_topics = max(log_likelihood_values, key=log_likelihood_values.get)
best_model = results[best_num_topics][0]
print(f"\nBest model based on log-likelihood is: {best_num_topics}-Topic model")

# Extract and print the top words from the best model
def print_top_words(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic #{topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

print("\nInterpreting Topics for the Best Model:")
print_top_words(best_model, vectorizer.get_feature_names_out())
