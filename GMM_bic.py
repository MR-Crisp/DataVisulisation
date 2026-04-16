import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
import pandas as pd

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
        bic_values = self.calculate_bic_for_gmm(X, max_clusters=20)
        optimal_clusters = np.argmin(bic_values) + 1
        gmm= GaussianMixture(n_components=optimal_clusters, covariance_type='full', random_state=42).fit(X)
        labels = gmm.predict(X)
        return labels,gmm

    def visual(self, X, labels, gmm):
        max_points = 5000
        if len(X) > max_points:
            idx = np.random.choice(len(X), max_points, replace=False)
            X_plot = X[idx]
            labels_plot = labels[idx]
        else:
            X_plot = X
            labels_plot = labels

        x_coords = X_plot[:, 0].tolist()
        y_coords = X_plot[:, 1].tolist()
        z_coords = X_plot[:, 2].tolist()
        labels_list = labels_plot.tolist()

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=4,
                color=labels_list,
                colorscale='Viridis',
                opacity=0.8,
                showscale=True,
                colorbar=dict(title='Cluster', x=1.0, len=0.5)
            ),
            text=[f'Cluster: {labels_list[i]}' for i in range(len(x_coords))],
            hoverinfo='text',
            name='Data Points'
        ))

        fig.add_trace(go.Scatter3d(
            x=gmm.means_[:, 0].tolist(),
            y=gmm.means_[:, 1].tolist(),
            z=gmm.means_[:, 2].tolist(),
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='diamond',
                line=dict(width=3, color='black'),
                opacity=1.0
            ),
            text=[f'Centroid {i}' for i in range(len(gmm.means_))],
            hoverinfo='text',
            name='Centroids'
        ))

        fig.update_layout(
            title='3D GMM Clustering of Latent Space',
            scene=dict(
                xaxis_title='Latent Dimension 1',
                yaxis_title='Latent Dimension 2',
                zaxis_title='Latent Dimension 3',
                xaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)',
                           gridcolor='white', showline=True, zerolinecolor='white'),
                yaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)',
                           gridcolor='white', showline=True, zerolinecolor='white'),
                zaxis=dict(showbackground=True, backgroundcolor='rgb(230, 230, 230)',
                           gridcolor='white', showline=True, zerolinecolor='white'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5),
                            center=dict(x=0, y=0, z=0),
                            up=dict(x=0, y=0, z=1)),
                aspectmode='cube'
            ),
            hovermode='closest',
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                        bgcolor='rgba(255, 255, 255, 0.8)'),
            margin=dict(l=0, r=0, b=0, t=40),
            updatemenus=[dict(
                type="buttons", direction="right", active=0, x=0.5, y=1.1,
                buttons=[
                    dict(label="Default 3D", method="relayout",
                         args=["scene.camera", dict(eye=dict(x=1.5, y=1.5, z=1.5))]),
                    dict(label="Top View", method="relayout",
                         args=["scene.camera", dict(eye=dict(x=0, y=0, z=3))]),
                    dict(label="Side View (X)", method="relayout",
                         args=["scene.camera", dict(eye=dict(x=3, y=0, z=0))]),
                    dict(label="Side View (Y)", method="relayout",
                         args=["scene.camera", dict(eye=dict(x=0, y=3, z=0))]),
                ]
            )]
        )

        return fig
