import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

class GMM():
    def __init__(self):
        pass
    # Function to calculate BIC for different numbers of clusters
    def calculate_bic_for_gmm(self,X, max_clusters:int):
        bic_values = []
        for n in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42).fit(X)
            bic_values.append(gmm.bic(X))
        return bic_values


    # Calculate BIC values for 1 to 10 clusters
    def GMM_calc(self,X):
        bic_values = self.calculate_bic_for_gmm(X, max_clusters=10)
        optimal_clusters = np.argmin(bic_values) + 1
        gmm= GaussianMixture(n_components=optimal_clusters, covariance_type='full', random_state=42).fit(X)
        labels = gmm.predict(X)
        return labels,gmm

    def visual(self,X,labels,gmm):
        plt.figure(figsize=(10, 8))
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', s=50, edgecolor='k')
        ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], gmm.means_[:, 2],
                   c='red', marker='X', s=300, label='Centroids')

        ax.set_title('Gaussian Mixture Model Clustering')
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.set_zlabel('Latent Dim 3')
        ax.legend()
        plt.show()
