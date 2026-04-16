
import torch
import torch.nn as nn
import torch.optim as optim

#from my files
from VAE import VariationalAutoencoder

def train_vae(model, train_loader, epochs=60, lr=0.001, beta_target=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-5)
    model.train()
    for epoch in range(epochs):
        annealing_factor = min(1.0, epoch / 10)
        beta = annealing_factor * beta_target  # ramps toward config beta, not 1.0
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimiser.zero_grad()
            recon_batch, mu, logvar = model(data)
            recon_loss = nn.functional.mse_loss(recon_batch, data, reduction='sum') / data.size(0)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + beta * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Total={avg_loss:.4f}, Recon={avg_recon_loss:.4f}, KL={avg_kl_loss:.4f}, Beta={beta:.4f}")

def get_tensor(df, target_col=None):
    if target_col and target_col in df.columns:
        X = df.drop(target_col, axis=1).values.astype('float32')
    else:
        X = df.values.astype('float32')
    return torch.tensor(X)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, input_dim, hidden_dim=128, latent_dim=3):
    model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(path, weights_only=False))
    model.eval()
    return model

# api_model.py — replace get_vae_config with this

def get_vae_config(D):
    types = D.types
    input_dim = D.X.shape[1]

    if "wide" in types or "complex" in types:
        latent_dim = 16
    elif "simple" in types or "small" in types:
        latent_dim = 3
    else:
        latent_dim = 8

    if "small" in types or "simple" in types:
        hidden_dim = 64
    elif "wide" in types or "complex" in types:
        hidden_dim = 512
    else:
        hidden_dim = 128

    if "simple" in types or "small" in types:
        n_layers = 2
    elif "complex" in types or "wide" in types:
        n_layers = 6
    else:
        n_layers = 4

    if "noisy" in types or "sparse" in types:
        beta = 0.1
    elif "simple" in types:
        beta = 0.005
    else:
        beta = 0.05

    dropout = 0.2 if "noisy" in types else 0.0

    config = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "n_layers": n_layers,
        "beta": beta,
        "dropout": dropout,
    }
    return config

