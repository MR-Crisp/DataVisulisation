import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt


from VAE import VariationalAutoencoder

# --- Data Loading and Preprocessing ---
df = pd.read_csv('taxi_zone_lookup.csv')

le_borough = LabelEncoder()
le_zone = LabelEncoder()
le_service = LabelEncoder()

df['Borough_enc'] = le_borough.fit_transform(df['Borough'])
df['Zone_enc'] = le_zone.fit_transform(df['Zone'])
df['service_zone_enc'] = le_service.fit_transform(df['service_zone'])
df['LocationID_scaled'] = df['LocationID']  # will be scaled below

# Scale ALL columns to [0, 1] so no single feature dominates the loss
features = df[['LocationID_scaled', 'Borough_enc', 'Zone_enc', 'service_zone_enc']].values
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
X = torch.tensor(features_scaled, dtype=torch.float32)

# --- Training ---
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = VariationalAutoencoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(500):
    total_loss = 0
    kl_weight = min(1.0, epoch / 100)  # slowly ramp up KL term

    for (batch,) in loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)

        recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_weight * kl_loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/500 | Loss: {total_loss/len(X):.4f}")

# --- Visualise Latent Space ---
model.eval()
with torch.no_grad():
    mu, _ = model.encode(X)
    mu = mu.numpy()

boroughs = df['Borough'].values
colours = {
    'Manhattan': 'red',
    'Brooklyn': 'blue',
    'Queens': 'green',
    'Bronx': 'orange',
    'Staten Island': 'purple',
    'EWR': 'grey'
}

plt.figure(figsize=(10, 7))
for borough, colour in colours.items():
    mask = boroughs == borough
    plt.scatter(mu[mask, 0], mu[mask, 1], c=colour, label=borough, alpha=0.7)

plt.legend()
plt.title("Taxi Zone VAE — Latent Space by Borough")
plt.xlabel("Latent dim 1")
plt.ylabel("Latent dim 2")
plt.savefig('latent_space.png', dpi=150, bbox_inches='tight')
print("Plot saved to latent_space.png")