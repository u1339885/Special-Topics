# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances_argmin_min

# Load the dataset
file_path = '/Users/marcwhiting/Library/Mobile Documents/com~apple~CloudDocs/Fall 2024/Chenglu/Week 6/Week 6 Dataset.csv'
data = pd.read_csv(file_path)


# Select the relevant columns for clustering
clustering_data = data[['message_qna', 'message_twitter', 'message_discussion']]

# Standardize the data to have a mean of 0 and variance of 1
scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

# Apply MiniBatchKMeans clustering (fast version of KMeans) with 10 clusters and max iterations set to 50
kmeans = MiniBatchKMeans(n_clusters=10, max_iter=50, batch_size=100, random_state=42)
kmeans.fit(clustering_data_scaled)

# Get the cluster assignments
cluster_labels = kmeans.labels_

# Add the cluster labels to the original dataframe
data['cluster'] = cluster_labels

# Count the number of people assigned to each cluster
cluster_counts = data['cluster'].value_counts()

# Output the number of clusters and cluster counts
num_clusters = len(cluster_counts)
print(f"Number of clusters identified: {num_clusters}")
print("\nNumber of people assigned to each cluster:")
print(cluster_counts)

# Get the centroids of each cluster
centroids = kmeans.cluster_centers_

# Create a DataFrame to display the centroids for each cluster
centroid_table = pd.DataFrame(centroids, columns=['message_qna', 'message_twitter', 'message_discussion'])

# Print the centroid table
print("Centroid Table:")
print(centroid_table)

# Plot the clusters along with their centroids (2D scatter plot)
plt.scatter(clustering_data_scaled[:, 0], clustering_data_scaled[:, 1], c=cluster_labels, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='x') # Mark the centroids
plt.title('Cluster Plot with Centroids (2D)')
plt.xlabel('message_qna')
plt.ylabel('message_twitter')
plt.show()

# Create a 3D scatter plot with the three variables and color by cluster
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot data points and color them based on their cluster
scatter = ax.scatter(clustering_data_scaled[:, 0], clustering_data_scaled[:, 1], clustering_data_scaled[:, 2],
                     c=cluster_labels, cmap='viridis', s=50)

# Add labels for the axes
ax.set_xlabel('message_qna')
ax.set_ylabel('message_twitter')
ax.set_zlabel('message_discussion')

# Add a color bar to show the mapping of colors to clusters
plt.colorbar(scatter, ax=ax, label='Cluster')

# Show the 3D plot
plt.title('3D Scatter Plot of Clustering Results')
plt.show()


# Calculate the distance of each point to its assigned centroid
_, distances = pairwise_distances_argmin_min(kmeans.cluster_centers_, clustering_data_scaled)

# Calculate the average within centroid distance (inertia or compactness) for the whole dataset
avg_within_centroid_distance = distances.mean()
print(f"Average within centroid distance (inertia): {avg_within_centroid_distance}")
