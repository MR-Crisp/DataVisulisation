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

def get_tensor(df):
    #Drop target column if exists
    if 'Cover_Type' in df.columns:
        X = df.drop('Cover_Type', axis=1).values.astype('float32')
    else:
        X = df.values.astype('float32')
    X_tensor = torch.tensor(X)
    return X_tensor

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, input_dim, hidden_dim=128, latent_dim=3):
    model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, weights_only=False))
    model.eval()
    return model

"""
#Load and preprocess the dataset
file_path = "./covertype.csv"
testset = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "zsinghrahulk/covertype-forest-cover-types",
    file_path
)

feature_cols = [col for col in testset.columns if col != 'Cover_Type']
target_col = 'Cover_Type'
dataset = StaticDataset(feature_cols=feature_cols, target_col=target_col)
dataset.input_dataset(testset)
dataset.preprocess() 

X_all = dataset.X
Y_all = dataset.Y

#Take sample of data for training
sample_size = int(0.1 * len(X_all))
X_tensor = torch.tensor(X_all[:sample_size].astype('float32'))
input_dim = X_tensor.shape[1]
vae_model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=128, latent_dim=3)

#Train/load/save the VAE model
# train_dataset = TensorDataset(X_tensor)
# train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
# train_vae(vae_model, train_loader, epochs=60, lr=0.001)
if os.path.exists('vae_model.pth'):
    print("Loading existing VAE model...")
    vae_model = load_model('vae_model.pth', input_dim=input_dim, hidden_dim=128, latent_dim=3)
else:
    print("Training new VAE model...")
    train_dataset = TensorDataset(X_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    train_vae(vae_model, train_loader, epochs=60, lr=0.001)
    save_model(vae_model, 'vae_model.pth')
    print("VAE model trained and saved as 'vae_model.pth'.")

#Encode to latent space
with torch.no_grad():
    mu, logvar = vae_model.encode(X_tensor)
    latent_vectors = mu.numpy()



#Visualisation options
choice = input("Do you want GMM or Voronoi(Type GMM or Vor): ")
if choice == "GMM":
    #Apply GMM clustering to the latent space
    gmm_model = GMM()
    labels, gmm = gmm_model.GMM_calc(latent_vectors)
    print(f"Number of clusters found: {len(np.unique(labels))}")
    print(f"GMM converged: {gmm.converged_}")
    print(f"Cluster distribution: {np.bincount(labels)}")
    gmm_model.visual(latent_vectors,labels, gmm)

elif choice == "Vor":
    # using latent to 2d using umap(dimesionality reduction)
    # Using full latent vectors (not just first 2 dims) gives a much more
    # meaningful layout — UMAP preserves local structure across all 3 dims.

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,  # local neighbourhood size — increase for smoother layout
        min_dist=0.1,  # how tightly points cluster — 0.0 = tightest
        random_state=42,
        metric='euclidean'
    )
    coords_2d = reducer.fit_transform(latent_vectors)  # shape (N, 2)

    #Pull Cover_Type labels aligned to the sample
    cover_labels = dataset.df['Cover_Type'].values[:sample_size].astype(int)
    unique_classes = np.unique(cover_labels)
    n_classes = len(unique_classes)

    # Build a colour map
    palette = px.colors.qualitative.Bold
    class_colour = {cls: palette[i % len(palette)] for i, cls in enumerate(unique_classes)}
    cover_type_names = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"}

    fig = plot_voronoi(coords_2d,cover_labels,class_colour,cover_type_names)

    fig.show()


"""