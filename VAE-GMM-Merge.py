import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 32, latent_dim = 2):#-----important the number WILL need to change, only done this for the current dataset, UPDATE has now been changed to 2 as make_blobs is only 2D
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim),nn.ReLU())#chaining input and hidden layers together
        #nn.Relu is the activation function
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)#this is for the mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)#this is for the variance in the normal distribution
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),nn.ReLU(),nn.Linear(hidden_dim, input_dim))
        """
        This is encoder but in reverse. First layer is connected expanding latent to hidden using ReLu
        second layer is for connection between hidden and 'input' layer and uses sigmoid to expand.
        sigmoid might need to be changed    
        """
    def encode(self, x):
        h = self.encoder(x)#passes x into input layer, and then h is returned from hidden layer
        return self.fc_mu(h), self.fc_logvar(h)# using h we then get the mean and variace (latent layer)

    def reparameterize(self, mu, logvar):# allows us to sample a latent space
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):#turns latent into hidden into 'output'(input)
        return self.decoder(z)

    def forward(self, x):#Glues the encoder and decoder, defines the path that the dat atakes from input to output
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


    def loss(self, recon_x, x, mu, logvar): #loss is reconstruction loss + KL divergence
        recon_loss = nn.functional.mse_loss(recon_x, x)#measures how well the output matches the input, this is the reconstruction loss
        kl_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp()) #measures how much the learned distribution deviates from a standard normal distribution, this is the KL divergence
        return recon_loss + kl_loss


    def train(self, X_tensor, epochs=200, lr=1e-3):
        optimiser = torch.optim.Adam(self.parameters(), lr=lr) # Use Adam optimiser to adjust leanrning rate
        self.train()
        for epoch in range(epochs): 
            optimiser.zero_grad() # Clear gradients before backpropagation
            recon, mu, logvar = self(X_tensor) # Forward pass
            loss = self.loss(recon, X_tensor, mu, logvar) # Calculate loss
            loss.backward() # Backpropagation to compute gradients
            optimiser.step() # Update model parameters

            if (epoch+1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.3f}') #Output loss every 50 epochs

    def get_latent_representation(self, X_tensor):
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(X_tensor) # Get mean and log variance from encoder
        return mu.numpy() # Return the mean as the latent representation

def calculate_bic(Z, max_clusters=10):
    bic_values = []
    for n in range(1, max_clusters + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42).fit(Z)
        bic_values.append(gmm.bic(Z))
    return bic_values


# Generate synthetic data
X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=[1.0,1.5,0.8], random_state=42)
X_tensor = torch.FloatTensor(X) # Convert to PyTorch tensor

#Train VAE model
model = VariationalAutoencoder(input_dim=2, hidden_dim=32, latent_dim=2) # Initialize VAE model
model.train(X_tensor, epochs=200)

#Get Latent Vectors
Z = model.get_latent_representation(X_tensor)

# Calculate BIC values for 1optimal clusters
bic_values = calculate_bic(Z, max_clusters=10)

# Determine the optimal number of clusters
optimal_clusters = np.argmin(bic_values) + 1
print(f'Optimal number of clusters according to BIC: {optimal_clusters}')


fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# Plot 1: raw data with true labels
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10', s=10)
axes[0].set_title('Raw Data (True Labels)')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# Plot 2: BIC curve
axes[1].plot(range(1, 11), bic_values, marker='o', color='steelblue')
axes[1].axvline(x=optimal_clusters, color='red', linestyle='--', label=f'Optimal={optimal_clusters}')
axes[1].set_title('BIC on Latent Space')
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('BIC')
axes[1].legend()
axes[1].grid(True)

# Plot 3: latent space with true labels
axes[2].scatter(Z[:, 0], Z[:, 1], c=y_true, cmap='tab10', s=10)
axes[2].set_title('Latent Space (True Labels)')
axes[2].set_xlabel('Latent Dim 1')
axes[2].set_ylabel('Latent Dim 2')

# Plot 4: latent space with GMM clusters + centers
axes[3].scatter(Z[:, 0], Z[:, 1], c=cluster_labels, cmap='tab10', s=10)
axes[3].scatter(gmm.means_[:, 0], gmm.means_[:, 1],
                c='red', marker='X', s=200, label='GMM Centers')
axes[3].set_title('Latent Space (GMM Clusters)')
axes[3].set_xlabel('Latent Dim 1')
axes[3].set_ylabel('Latent Dim 2')
axes[3].legend()

plt.tight_layout()
plt.show()

# --- Score ---
ari = adjusted_rand_score(y_true, cluster_labels)
print(f'Adjusted Rand Index: {ari:.3f}  (1.0 = perfect match)')


"""
# Plot BIC values
plt.plot(range(1, 11), bic_values, marker='o')
plt.title('BIC Values for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('BIC')
plt.show()

"""