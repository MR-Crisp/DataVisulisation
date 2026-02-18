import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate synthetic data
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=[1.0,1.5,0.8,1.2], random_state=42)
# Function to calculate BIC for different numbers of clusters
def calculate_bic_for_gmm(X, max_clusters:int):
    bic_values = []
    for n in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42).fit(X)
        bic_values.append(gmm.bic(X))
    return bic_values


# Calculate BIC values for 1 to 10 clusters
def GMM(X):
    bic_values = calculate_bic_for_gmm(X, max_clusters=10)
    optimal_clusters = np.argmin(bic_values) + 1
    gmm= GaussianMixture(n_components=optimal_clusters, covariance_type='full', random_state=42).fit(X)
    labels = gmm.predict(X)
    return labels,gmm

def visual(labels,gmm):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, edgecolor='k')
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='X', s=300, label='Centroids')
    plt.title('Gaussian Mixture Model Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True)
    plt.legend()
    plt.show()
label,gmm = GMM(X)
visual(label,gmm)