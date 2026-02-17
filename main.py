import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
from VAE import VariationalAutoencoder

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
        #remove all empty row/col
        df = df.dropna(how='all',axis=0)
        df = df.dropna(how='all',axis=1)
        #remove duplicate records
        df = df.drop_duplicates()
        # Drop rows where >50% of values are missing
        threshold = len(df.columns) * 0.5
        df = df.dropna(thresh=threshold, axis=0)
        self.df = df
        print(df.isnull().sum())


testset = "./taxi_zone_lookup.csv"

D = StaticDataset()
D.input_dataset(testset)
D.clean_dataset()
print(D.df.shape)


def train_vae(model, train_loader, epochs=100, lr=0.001):
    optimiser = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            optimiser.zero_grad()
            
            # Forward pass using YOUR VAE's forward method
            recon_batch, mu, logvar = model(data)
            
            # VAE Loss
            recon_loss = nn.MSELoss(reduction='sum')(recon_batch, data)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            
            loss.backward()
            optimiser.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Loss = {total_loss/len(train_loader.dataset):.4f}')

