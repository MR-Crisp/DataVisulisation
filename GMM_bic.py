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
        bic_values = self.calculate_bic_for_gmm(X, max_clusters=20)
        optimal_clusters = np.argmin(bic_values) + 1
        gmm= GaussianMixture(n_components=optimal_clusters, covariance_type='full', random_state=42).fit(X)
        labels = gmm.predict(X)
        return labels,gmm

    def visual(self,X,labels,gmm):
        print("Starting visualization...")
        print(f"Data shape: {X.shape}")
        print(f"Number of clusters: {len(np.unique(labels))}")
        print(f"Centroid coordinates:\n{gmm.means_}")
        x_coords = X[:, 0]
        y_coords = X[:, 1]
        z_coords = X[:, 2]
        print(f"X range: [{x_coords.min():.3f}, {x_coords.max():.3f}]")
        print(f"Y range: [{y_coords.min():.3f}, {y_coords.max():.3f}]")
        print(f"Z range: [{z_coords.min():.3f}, {z_coords.max():.3f}]")

        fig = go.Figure()

        # Add data points with cluster coloring
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers', 
            marker=dict(
                size=4,
                color=labels, 
                colorscale='Viridis', 
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title='Cluster',
                    x=1.0,
                    len=0.5
                )
            ),
            text=[f'Point {i}<br>Cluster: {labels[i]}<br>Coordinates: ({x_coords[i]:.3f}, {y_coords[i]:.3f}, {z_coords[i]:.3f})' for i in range(len(X))],
            hoverinfo='text',
            name='Data Points'
        ))

        # Add cluster centroids
        fig.add_trace(go.Scatter3d(
            x=gmm.means_[:, 0], 
            y=gmm.means_[:, 1],
            z=gmm.means_[:, 2], 
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
                xaxis=dict(
                    showbackground=True,
                    backgroundcolor='rgb(230, 230, 230)',
                    gridcolor='white',
                    showline=True,
                    zerolinecolor='white',
                ),
                yaxis=dict(
                    showbackground=True,
                    backgroundcolor='rgb(230, 230, 230)',
                    gridcolor='white',
                    showline=True,
                    zerolinecolor='white',
                ),
                zaxis=dict(
                    showbackground=True,
                    backgroundcolor='rgb(230, 230, 230)',
                    gridcolor='white',
                    showline=True,
                    zerolinecolor='white',
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='cube'
            ),
            width=1000,
            height=800,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.5,
                    y=1.1,
                    buttons=list([
                        dict(
                        label="Default 3D",
                        method="relayout",
                        args=["scene.camera", dict(eye=dict(x=1.5, y=1.5, z=1.5))]
                        ),
                        dict(
                            label="Top View",
                            method="relayout",
                            args=["scene.camera", dict(eye=dict(x=0, y=0, z=3))]
                        ),
                        dict(
                            label="Side View (X)",
                            method="relayout",
                            args=["scene.camera", dict(eye=dict(x=3, y=0, z=0))]
                        ),
                        dict(
                            label="Side View (Y)", 
                            method="relayout",
                            args=["scene.camera", dict(eye=dict(x=0, y=3, z=0))]
                        ),
                        dict(
                            label="Rotate",
                            method="animate",
                            args=[None, dict(
                                frame=dict(duration=50, redraw=True),
                                fromcurrent=True,
                                mode='immediate',
                                transition=dict(duration=0)
                            )]
                        )
                    ])
                )
            ]
        )

        def create_rotation_frames():
            frames = []
            for angle in range(0, 360, 5):
                frames.append(go.Frame(
                    layout=dict(
                        scene=dict(
                            camera=dict(
                                eye=dict(
                                    x=1.5 * np.cos(np.radians(angle)), 
                                    y=1.5 * np.sin(np.radians(angle)), 
                                    z=1
                                )
                            )
                                
                        )
                    )
                )) 
            return frames

        fig.frames = create_rotation_frames()
        fig.show()
