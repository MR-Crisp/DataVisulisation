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

