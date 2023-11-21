#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:53:00 2023

@author: ramrajvemuri
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

# Load the glass dataset
glass_data = pd.read_csv('/Users/ramrajvemuri/Downloads/glass.csv')

# Convert 'Type' to a binary classification target
glass_data['Binary_Type'] = glass_data['Type'].apply(lambda x: 1 if x == 1 else 0)

# Separate features and target
X_glass = glass_data.drop(['Type', 'Binary_Type'], axis=1)
y_glass = glass_data['Binary_Type']

# Split the dataset into training and testing sets
X_train_glass, X_test_glass, y_train_glass, y_test_glass = train_test_split(X_glass, y_glass, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_glass_scaled = scaler.fit_transform(X_train_glass)
X_test_glass_scaled = scaler.transform(X_test_glass)

# Train a logistic regression model
lr_glass = LogisticRegression(random_state=42)
lr_glass.fit(X_train_glass_scaled, y_train_glass)

# Get predicted probabilities
y_probs_glass = lr_glass.predict_proba(X_test_glass_scaled)[:, 1]  # Get the probability of the positive class

# Function to calculate accuracy, precision and recall for different thresholds
def evaluate_threshold(threshold):
    y_pred_glass = (y_probs_glass >= threshold).astype(int)
    accuracy = accuracy_score(y_test_glass, y_pred_glass)
    precision = precision_score(y_test_glass, y_pred_glass)
    recall = recall_score(y_test_glass, y_pred_glass)
    return accuracy, precision, recall

# Evaluate the model for different thresholds
thresholds = [0.4, 0.5, 0.6, 0.7]
evaluation_metrics = {t: evaluate_threshold(t) for t in thresholds}

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test_glass, y_probs_glass)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Glass Dataset')
plt.legend(loc="lower right")
plt.show()

# Load the iris dataset
iris_data = pd.read_csv('/Users/ramrajvemuri/Downloads/iris.csv')  # Make sure to use the correct path to your dataset

# Separate features from the iris dataset
X_iris = iris_data.drop(['Name'], axis=1)

# List to store the results
inertia_scores = []
silhouette_scores = []

# Range of k values to try
k_values = range(2, 11)

# Clustering with different values of k and without scaling features
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_iris)
    
    # Inertia: Sum of distances of samples to their closest cluster center
    inertia_scores.append(kmeans.inertia_)
    
    # Silhouette Score: Mean silhouette coefficient over all samples
    silhouette_scores.append(silhouette_score(X_iris, kmeans.labels_))

# Now let's repeat the clustering with scaled features
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# List to store the results for scaled data
inertia_scores_scaled = []
silhouette_scores_scaled = []

# Clustering with different values of k and with scaling features
for k in k_values:
    kmeans_scaled = KMeans(n_clusters=k, random_state=42)
    kmeans_scaled.fit(X_iris_scaled)
    
    inertia_scores_scaled.append(kmeans_scaled.inertia_)
    silhouette_scores_scaled.append(silhouette_score(X_iris_scaled, kmeans_scaled.labels_))

# Plotting the results for unscaled data
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_scores, marker='o', linestyle='-', label='Inertia')
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', label='Silhouette Score')
plt.title('Clustering Metrics for Different k (Unscaled Data)')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Score')
plt.legend()

# Plotting the results for scaled data
plt.subplot(1, 2, 2)
plt.plot(k_values, inertia_scores_scaled, marker='o', linestyle='-', label='Inertia')
plt.plot(k_values, silhouette_scores_scaled, marker='o', linestyle='--', label='Silhouette Score')
plt.title('Clustering Metrics with Scaled Features for Different k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Score')
plt.legend()

plt.tight_layout()
plt.show()