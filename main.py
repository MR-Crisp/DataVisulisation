import numpy as np
import matplotlib.pyplot as plt
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



#Use normalised data
X_tensor = get_tensor(D.df)
sample_size = int(0.1 * len(X_tensor))  # Use 10% of the data for training
X_tensor = X_tensor[:sample_size]  # Take the first 10% of the data for training

dataset = TensorDataset(X_tensor)
train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

#Get input dimension for VAE
input_dim = X_tensor.shape[1]

vae = VariationalAutoencoder(input_dim=input_dim, hidden_dim=128, latent_dim=3)
train_vae(vae,train_loader,200, lr=0.001)

vae.eval()#eval inherited from nn module
with torch.no_grad():
    mu, logvar = vae.encode(X_tensor)
    latent_vectors = mu.numpy()

#Apply GMM clustering to the latent space
gmm_model = GMM()
labels, gmm = gmm_model.GMM_calc(latent_vectors)
gmm_model.visual(latent_vectors,labels, gmm)