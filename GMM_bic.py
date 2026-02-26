import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture

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
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X[:, 0], 
            y=X[:, 1], 
            mode='markers', 
            marker=dict(
                size=3,
                color=labels, 
                colorscale='Viridis', 
                opacity=0.8,
                showscale=True,
                colorbar=dict(title='Cluster')
            ),
            text=[f'Point {i}<br>Cluster: {labels[i]}' for i in range(len(labels))],
            hoverinfo='text',
            name='Data Points'
        ))
        
        fig.add_trace(go.Scatter(
            x=gmm.means_[:, 0], 
            y=gmm.means_[:, 1], 
            mode='markers', 
            marker=dict(
                size=10,
                color='red', 
                symbol='x',
                line=dict(width=2, color='black')
            ),
            text=[f'Centroid {i}' for i in range(len(gmm.means_))],
            hoverinfo='text',
            name='Centroids'
        ))

        fig.update_layout(
            title='GMM Clustering of Latent Space',
            xaxis_title='Latent Dimension 1',
            yaxis_title='Latent Dimension 2',
            zaxis_title='Latent Dimension 3',
            legend_title='Legend',
            width=900,
            height=700
        )
        fig.show()
