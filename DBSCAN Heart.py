# Source Code Title: DBSCAN Heart
# Source Code Author: Joshua Ryan Steenson
# [CSCI]-[594]: Clustering Techniques Final Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gower

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


heart = pd.read_csv('heart.csv')
print("Shape of the Data Frame: ", heart.shape, "\n")

############## Preprocessing Step

# Build the scaler model
scaler = MinMaxScaler()

# Fit the training data set
scaler.fit(heart)

# Transform the test data set
heart_scaled = scaler.transform(heart)

# Verify minimum value of all features
heart_scaled.min(axis=0)

# Verify maximum value of all features
heart_scaled.max(axis=0)

# Manually normalize without utilizing scikit-learn
heart_manual_scaled = (heart-heart.min(axis=0))/(heart.max(axis=0)-heart.min(axis=0))

# Verify manually versus scikit-learn estimation
print(np.allclose(heart_scaled, heart_manual_scaled), "\n") # Prints TRUE

pca = PCA(n_components = 2)
heart_principal = pca.fit_transform(heart_scaled)
heart_principal = pd.DataFrame(heart_principal)
heart_principal.columns = ['P1', 'P2']

############## Compute DBSCAN via the Euclidean distance

clustering_euclidean = DBSCAN(eps=0.5, min_samples=30).fit(heart_principal)
core_samples_mask = np.zeros_like(clustering_euclidean.labels_, dtype=bool)
core_samples_mask[clustering_euclidean.core_sample_indices_] = True
labels = clustering_euclidean.labels_

# Number of clusters in labels, ignoring noise if present
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("**DBSCAN Euclidean 13-Feature Performance Evaluation**\n")
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(heart_principal, labels))
print()

# Building the label to color mapping
colors = {}
colors[0] = 'r'
colors[1] = 'g'
colors[2] = 'b'
colors[3] = 'c'
colors[4] = 'y'
colors[5] = 'm'
colors[-1] = 'k'

# Building the color vector for each data point
cvec = [colors[label_one] for label_one in labels]
colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

r = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[0]);
g = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[1]);
b = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[2]);
c = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[3]);
y = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[4]);
m = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[5]);
k = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[6]);

# Plotting P1 on the x-axis and P2 on the y-axis according to the color vector defined
plt.figure(figsize=(9,9))
plt.scatter(heart_principal['P1'], heart_principal['P2'], c = cvec)
# Building the legend
plt.legend((r, g, b, c, y, m, k),
           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'),
           scatterpoints = 1, loc = 'upper left', ncol=3, fontsize=8)
plt.show()

############## Compute DBSCAN via the Euclidean distance with the uppermost two features

heart_two = heart[["ejection_fraction", "serum_creatinine"]]

scaler_two = MinMaxScaler()
scaler_two.fit(heart_two)
heart_two = scaler_two.transform(heart_two)
heart_two.min(axis=0)
heart_two.max(axis=0)
heart_manual_two = (heart_two-heart_two.min(axis=0))/(heart_two.max(axis=0)-heart_two.min(axis=0))

print(np.allclose(heart_two, heart_manual_two), "\n") # Prints TRUE

pca_2 = PCA(n_components = 2)
heart_principal_two = pca_2.fit_transform(heart_two)
heart_principal_two = pd.DataFrame(heart_principal_two)
heart_principal_two.columns = ['P1', 'P2']

dbs_two = DBSCAN(eps=0.5, min_samples=55).fit(heart_principal_two)
labels_two = dbs_two.labels_
core_samples_mask = np.zeros_like(dbs_two.labels_, dtype=bool)
core_samples_mask[dbs_two.core_sample_indices_] = True
n_clusters_two_ = len(set(labels_two)) - (1 if -1 in labels_two else 0)
n_noise_two_ = list(labels_two).count(-1)

print("**DBSCAN Euclidean Two-Feature Performance Evaluation**\n")
print('Estimated number of clusters: %d' % n_clusters_two_)
print('Estimated number of noise points: %d' % n_noise_two_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(heart_principal_two, labels_two))
print()

# Building the label to color mapping
colors2 = {}
colors2[0] = 'r'
colors2[1] = 'g'
colors2[2] = 'b'
colors2[3] = 'c'
colors2[4] = 'y'
colors2[5] = 'm'
colors2[-1] = 'k'

# Building the color vector for each data point
cvec = [colors2[label_two] for label_two in labels_two]
colors = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

r = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[0]);
g = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[1]);
b = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[2]);
c = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[3]);
y = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[4]);
m = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[5]);
k = plt.scatter(heart_principal['P1'], heart_principal['P2'], marker = 'o', color = colors[6]);

# Plotting P1 on the x-axis and P2 on the y-axis according to the color vector defined
plt.figure(figsize=(9,9))
plt.scatter(heart_principal_two['P1'], heart_principal_two['P2'], c = cvec)
# Building the legend
plt.legend((r, g, b, c, y, m, k),
           ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'),
           scatterpoints = 1, loc = 'upper left', ncol=3, fontsize=8)
plt.show()

############## Compute DBSCAN via the Gower distance

GowerDistance = gower.gower_matrix(heart_scaled)
clustering_gower = DBSCAN(eps=0.5, min_samples=10).fit(GowerDistance)
core_samples_mask = np.zeros_like(clustering_gower.labels_, dtype=bool)
core_samples_mask[clustering_gower.core_sample_indices_] = True
labels_gower = clustering_gower.labels_

n_clusters_gower_ = len(set(labels_gower)) - (1 if -1 in labels_gower else 0)
n_noise_gower_ = list(labels_gower).count(-1)

print("**DBSCAN Gower 13-Feature Performance Evaluation**\n")
print('Estimated number of clusters: %d' % n_clusters_gower_)
print('Estimated number of noise points: %d' % n_noise_gower_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(GowerDistance, labels))
print()

# The following hyperlinks are the sources that were utilized to develop the source code. Many of them are cited in the paper.
###########################################################################################################################################
# https://www.geeksforgeeks.org/implementing-dbscan-algorithm-using-sklearn/
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
# https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
# https://medium.com/analytics-vidhya/concept-of-gowers-distance-and-it-s-application-using-python-b08cf6139ac2
# https://towardsdatascience.com/dbscan-algorithm-complete-guide-and-application-with-python-scikit-learn-d690cbae4c5d
# https://towardsdatascience.com/everything-you-need-to-know-about-min-max-normalization-in-python-b79592732b79
# https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
############################################################################################################################################
