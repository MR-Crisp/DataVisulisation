import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import Voronoi
import umap

#from my files
from VAE import VariationalAutoencoder
from GMM_bic import GMM
from voronoi_algorithm import voronoi_finite_polygons,plot_voronoi
from Dataset import StaticDataset

def train_vae(model, train_loader, epochs=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch_idx, (data,) in enumerate(train_loader):#for every epoch, go through the batch and optimise weights
            data = data.to(device)
            optimiser.zero_grad()#optimiser
            recon_batch, mu, logvar = model(data)#forward pass

            # VAE Loss using mean square error
            recon_loss = nn.MSELoss(reduction='sum')(recon_batch, data)#loss of input vs output (encoder)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())#comparing variance vs normal distribution N(0,1)

            beta = 0.1 #Beta-VAE approach to balance reconstruction and KL divergence
            loss = recon_loss + beta * kl_loss#total loss

            loss.backward()#backwards pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)#gradient clipping to prevent exploding gradients
            optimiser.step()#next
            #Updating the loss
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        if epoch % 20 == 0:#for every 20 epochs
            avg_loss = total_loss / len(train_loader.dataset)
            avg_recon_loss = total_recon_loss / len(train_loader.dataset)
            avg_kl_loss = total_kl_loss / len(train_loader.dataset)

            print(f'Epoch {epoch}: Total loss = {avg_loss:.4f}, Recon Loss = {avg_recon_loss:.4f}, KL Loss = {avg_kl_loss:.4f}')

def get_tensor(df, target_col=None):
    if target_col and target_col in df.columns:
        X = df.drop(target_col, axis=1).values.astype('float32')
    else:
        X = df.values.astype('float32')
    return torch.tensor(X)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, input_dim, hidden_dim=128, latent_dim=3):
    model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, weights_only=False))
    model.eval()
    return model

# api_model.py — replace get_vae_config with this

def get_vae_config(D):
    types = D.types
    input_dim = D.X.shape[1]
    latent_dim = 3

    if "small" in types or "simple" in types:# setting hidden dimension block
        hidden_dim = 64
    elif "wide" in types or "complex" in types:
        hidden_dim = 512
    else:
        hidden_dim = 128

    if "simple" in types or "small" in types:#hidden layers
        n_layers = 2
    elif "complex" in types or "wide" in types:
        n_layers = 6
    else:
        n_layers = 4

    if "noisy" in types or "sparse" in types:#beta setter
        beta = 0.1
    elif "simple" in types:
        beta = 0.005
    else:
        beta = 0.05

    dropout = 0.2 if "noisy" in types else 0.0 # dropout only if noisy

    config = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "n_layers": n_layers,
        "beta": beta,
        "dropout": dropout,
    }
    return config