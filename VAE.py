import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim = 4, hidden_dim = 128, latent_dim = 2, beta = 0.1, dropout = 0.0, n_layers = 2):#defaults but will be changed
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta

        #building encoder dynamically
        in_dim = self.input_dim
        encoder_layers = []
        for i in range(n_layers):
            out_dim = hidden_dim if  i< n_layers-1 else hidden_dim//2
            encoder_layers+=[nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*encoder_layers)
        encoder_out_dim = hidden_dim // 2

        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)

        #mirror the encoder
        decoder_layers = []
        in_dim = latent_dim
        dims = [hidden_dim // 2] + [hidden_dim] * (n_layers - 1)
        for out_dim in dims:
            decoder_layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
            ]
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

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

