import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic data
X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=[1.0,1.5,0.8], random_state=42)
# Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)
# Predict the cluster labels
labels = gmm.predict(X)
# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, edgecolor='k')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='X', s=300, label='Centroids')
plt.title('Gaussian Mixture Model Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.legend()
plt.show()