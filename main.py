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

#from my files
from VAE import VariationalAutoencoder
from GMM_bic import GMM
from summary_algorithm import summaryAlgorithm
from summary_report import SummaryReport



class StaticDataset:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()

    def input_covertype_dataset(self,location):
        df = location
        self.df = df

    def clean_covertype_dataset(self):
        df = self.df.copy()

        #Remove unnamed/index columns
        unnamed_cols = [col for col in df.columns if 'unnamed' in col.lower() or 'index' in col.lower()]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        
        #Remove all empty row/columns
        df = df.dropna(how='all', axis=0)  # Drop rows
        df = df.dropna(how='all', axis=1)  # Drop columns

        #Remove duplicates
        df = df.drop_duplicates()

        #Drop rows where >50% of values are missing
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold, axis=0)

        self.df = df


    def normalise_covertype_data(self):
        #Seperate features and target if needed
        if 'Cover_Type' in self.df.columns:
            features = self.df.drop('Cover_Type', axis=1)
            target = self.df['Cover_Type']

            #Normalise features
            normalised_features = self.scaler.fit_transform(features)

            #Combine normalised features with target
            self.df = pd.DataFrame(normalised_features, columns=features.columns)
            self.df['Cover_Type'] = target.values
        else:
            #Normalise all data if no target column
            self.df = pd.DataFrame(self.scaler.fit_transform(self.df), columns=self.df.columns)

        return self.df

def train_vae(model, train_loader, epochs=100, lr=0.001):
    optimiser = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch_idx, (data,) in enumerate(train_loader):#for every epoch, go through the batch and optimise weights
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



file_path = "./covertype.csv"
testset = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "zsinghrahulk/covertype-forest-cover-types",
    file_path
)

D = StaticDataset()
D.input_covertype_dataset(testset)
D.clean_covertype_dataset()
D.normalise_covertype_data()
print(D.df.shape)

input_dim = D.df.shape[1] - 1 if 'Cover_Type' in D.df.columns else D.df.shape[1]
X_tensor = get_tensor(D.df)
sample_size = int(0.1 * len(X_tensor))
X_tensor = X_tensor[:sample_size]

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
    print("VAE model trained and saved as 'vae_model.pth'.")

with torch.no_grad():
    mu, logvar = vae_model.encode(X_tensor)
    latent_vectors = mu.numpy()

gmm_model = GMM()
labels, gmm = gmm_model.GMM_calc(latent_vectors)
print(f"Number of clusters found: {len(np.unique(labels))}")
print(f"GMM converged: {gmm.converged_}")
print(f"Cluster distribution: {np.bincount(labels)}")

summary_model = summaryAlgorithm()
cluster_summary = summary_model.summarise_clusters(latent_vectors, labels, gmm)

print("\nCluster Summary:")
print(cluster_summary)

text_summaries = summary_model.generate_text_summary(cluster_summary)
print("\nText Summary:")
for summary in text_summaries:
    print(summary)

overall_summary = summary_model.generate_overall_summary(cluster_summary, gmm)
print("\nOverall Summary:")
for summary in overall_summary:
    print(summary)



gmm_model.visual(latent_vectors, labels, gmm)
