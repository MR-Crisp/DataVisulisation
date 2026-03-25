import numpy as np
import pandas as pd
import torch
import os
import base64
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import kagglehub
from kagglehub import KaggleDatasetAdapter

from VAE import VariationalAutoencoder
from GMM_bic import GMM
from main import StaticDataset, get_tensor, load_model, train_vae, save_model


file_path = "./covertype.csv"
testset = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS, "zsinghrahulk/covertype-forest-cover-types", file_path)

D = StaticDataset()
D.input_covertype_dataset(testset)
D.clean_covertype_dataset()
D.normalise_covertype_data()
print(f"Dataset shape: {D.df.shape}")

input_dim = D.df.shape[1] - 1 if 'Cover_Type' in D.df.columns else D.df.shape[1]
X_tensor = get_tensor(D.df)
sample_size = int(0.1 * len(X_tensor))
X_tensor = X_tensor[:sample_size]

# Load or train VAE
if os.path.exists('vae_model.pth'):
    print("Loading existing VAE model...")
    vae_model = load_model('vae_model.pth', input_dim=input_dim, hidden_dim=128, latent_dim=3)
else:
    print("Training new VAE model...")
    dataset = TensorDataset(X_tensor)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    vae_model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=128, latent_dim=3)
    train_vae(vae_model, train_loader, epochs=60, lr=0.001)
    save_model(vae_model, 'vae_model.pth')

# Encode data to latent space
with torch.no_grad():
    mu, logvar = vae_model.encode(X_tensor)
    latent_vectors = mu.numpy()

# Fit GMM (with BIC)
gmm_model = GMM()
labels, gmm = gmm_model.GMM_calc(latent_vectors)
print(f"Number of clusters: {len(np.unique(labels))}")

# Feature names (excluding target column)
feature_names = [col for col in D.df.columns if col != 'Cover_Type']
n_features = len(feature_names)

def decode_latent(z):
    with torch.no_grad():
        z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
        reconstructed = vae_model.decode(z_tensor).numpy().flatten()
    return reconstructed

def create_soft_3d_plot():
    probs = gmm.predict_proba(latent_vectors)
    n_clusters = probs.shape[1]
    # Get cluster colours (tab10 works for up to 10 clusters)
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    cluster_colors = cmap(np.arange(n_clusters))[:, :3]   # (n_clusters, 3)
    point_colors = probs @ cluster_colors                # (n_samples, 3)
    point_colors_hex = [f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for c in point_colors]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=latent_vectors[:, 0],
        y=latent_vectors[:, 1],
        z=latent_vectors[:, 2],
        mode='markers',
        marker=dict(size=4, color=point_colors_hex, opacity=0.8),
        text=[f'Point {i}<br>Probabilities: {probs[i]}' for i in range(len(latent_vectors))],
        hoverinfo='text',
        name='Data points',
        customdata=latent_vectors
    ))
    # Add centroids
    fig.add_trace(go.Scatter3d(
        x=gmm.means_[:, 0],
        y=gmm.means_[:, 1],
        z=gmm.means_[:, 2],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='black')),
        name='Centroids'
    ))
    # Add a trace for the selected point (initially invisible)
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=15, color='yellow', symbol='circle', line=dict(width=2, color='black')),
        name='Selected point'
    ))
    fig.update_layout(
        title='Latent Space Explorer – Click on any point or use sliders',
        scene=dict(
            xaxis_title='Latent dim 1',
            yaxis_title='Latent dim 2',
            zaxis_title='Latent dim 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        ),
        width=800, height=600
    )
    return fig

def heatmap_from_features(features):
    """Generate a stylised heatmap image (base64) from a feature vector"""
    fig, ax = plt.subplots(figsize=(12, 8))
    # Reshape 52 features into 13x4 grid
    reshaped = features.reshape(13, 4) if len(features) == 52 else features.reshape(1, -1)
    im = ax.imshow(reshaped, cmap='RdYlBu', aspect='auto', interpolation='bilinear')
    ax.set_xticks(np.arange(reshaped.shape[1]))
    ax.set_yticks(np.arange(reshaped.shape[0]))
    if len(features) == 52:
        ax.set_xticklabels([f'F{i}' for i in range(4)])
        ax.set_yticklabels([f'Group {i}' for i in range(13)])
    else:
        ax.set_xticklabels([f'F{i}' for i in range(len(features))], rotation=90)
        ax.set_yticks([])
    plt.colorbar(im, ax=ax, label='Feature value')
    ax.set_title('Generated Sample – Feature Heatmap')
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{encoded}'

