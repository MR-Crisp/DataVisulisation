import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
import pandas as pd
#from my files

from VAE import VariationalAutoencoder
from GMM_bic import GMM




class StaticDataset:
    def __init__(self):
        self.df = None

    def input_dataset(self,location):
        df = pd.read_csv(location)
        self.df = df
        # print(df.head())
        # print(df.columns)
        # print(df.shape)
        print(df.isnull().sum())

    def clean_dataset(self):
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


testset = "./covertype.csv"

D = StaticDataset()
D.input_dataset(testset)
D.clean_dataset()
print(D.df.shape)


def train_vae(model, train_loader, epochs=100, lr=0.001):
    optimiser = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):#for every epoch, go through the batch and optimise weights
            optimiser.zero_grad()#optimiser
            recon_batch, mu, logvar = model(data)#forward pass
            
            # VAE Loss using mean square error
            recon_loss = nn.MSELoss(reduction='sum')(recon_batch, data)#loss of input vs output (encoder)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())#comparing variance vs normal distribution N(0,1)
            loss = recon_loss + kl_loss#total loss
            
            loss.backward()#backwards pass
            optimiser.step()#next
            total_loss += loss.item()#updating the loss
        
        if epoch % 20 == 0:#for every 20 epochs
            print(f'Epoch {epoch}: Loss = {total_loss/len(train_loader.dataset):.4f}')# print loss


def prepare_data(df):
    # Drop columns that are not useful for clustering
    drop_cols = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime',  # raw timestamps
        'extra', 'mta_tax', 'improvement_surcharge',       # fixed fee columns, low signal
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Drop rows with nulls
    df = df.dropna()

    # Label encode categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df

def get_tensor(df):
    X = df.values.astype('float32')
    X_tensor = torch.tensor(X)
    return X_tensor

df = pd.read_csv('./covertype.csv')

# df_clean = prepare_data(df)
df_clean = df.dropna()#drop rows with nulls

X_tensor = get_tensor(df_clean)
dataset = TensorDataset(X_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = df_clean.shape[1]
vae = VariationalAutoencoder(input_dim=input_dim, hidden_dim=32, latent_dim=2)
train_vae(vae,train_loader,500, lr=0.01)
vae.eval()#eval inherited from nn module
with torch.no_grad():#lets me speed things up
    mu, logvar = vae.encode(X_tensor)
latent_vectors = mu.numpy()
gmm_model = GMM()
labels, gmm = gmm_model.GMM_calc(latent_vectors)
gmm_model.visual(latent_vectors,labels, gmm)