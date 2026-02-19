import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim = 128, latent_dim = 2):#-----important the number WILL need to change, only done this for the current dataset
        super().__init__()
        # Added more layers to the encoder to increase its capacity to learn complex representations. 
        # Batch normalization is included to stabilize training and improve convergence.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        encoder_outut_dim = hidden_dim // 2

        self.fc_mu = nn.Linear(encoder_outut_dim, latent_dim)#this is for the mean
        self.fc_logvar = nn.Linear(encoder_outut_dim, latent_dim)#this is for the variance in the normal distribution
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
            )

    def encode(self, x):
        h = self.encoder(x)#passes x into input layer, and then h is returned from hidden layer
        return self.fc_mu(h), self.fc_logvar(h)# using h we then get the mean and variace (latent layer)

    def reparameterize(self, mu, logvar):# allows us to sample a latent space
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):#turns latent into hidden into 'output'(input)
        return self.decoder(z)

    def forward(self, x):#didnt understand
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

