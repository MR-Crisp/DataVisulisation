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

