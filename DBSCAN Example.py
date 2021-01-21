import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_iris

iris_data = load_iris()

X = iris_data['data']

print(X.shape)

labels_iris = iris_data['target']

print(labels_iris)

n_true_clusters = len(set(labels_iris))

print(n_true_clusters)

dbs = DBSCAN(eps=0.3, min_samples=5)

print(dbs.fit(X))

print(dbs.labels_)

set(dbs.labels_)

n_clusters = len(set(dbs.labels_)) - (1 if -1 in dbs.labels_ else 0)

print(n_clusters)

n_noise_points = sum(dbs.labels_ == -1)

print(n_noise_points)

dbs = DBSCAN(eps=0.5, min_samples=5)

print(dbs.fit(X))

print(dbs.labels_)

n_clusters = len(set(dbs.labels_)) - (1 if -1 in dbs.labels_ else 0)

print(n_clusters)

n_noise_points = sum(dbs.labels_ == -1)

print(n_noise_points)

dbs = DBSCAN(eps=0.5, min_samples=4)
print(dbs.fit(X))
n_clusters = len(set(dbs.labels_)) - (1 if -1 in dbs.labels_ else 0)
print(n_clusters)
n_noise_points = sum(dbs.labels_ == -1)
print(n_noise_points)
