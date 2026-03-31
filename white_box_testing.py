import pytest
import torch
from VAE import VariationalAutoencoder

def test_VAE_init():
    vae = VariationalAutoencoder(input_dim=4, hidden_dim=128, latent_dim=2)
    assert vae.input_dim == 4
    assert vae.hidden_dim == 128
    assert vae.latent_dim == 2
    assert vae.encoder is not None
    assert vae.decoder is not None

# VAE-02: encode correct shape
def test_VAE_encode():
    vae = VariationalAutoencoder(input_dim=4, hidden_dim=128, latent_dim=3)
    x = torch.randn(5, 4)
    mu, logvar = vae.encode(x)
    assert mu.shape == (5, 3)
    assert logvar.shape == (5, 3)

# VAE-03: encode edge case
def test_VAE_encode_edge():
    vae = VariationalAutoencoder(input_dim=4, hidden_dim=128, latent_dim=3)
    x = torch.randn(5, 3)
    with pytest.raises(RuntimeError):
        vae.encode(x)

# VAE-04: reparameterize correct shape
def test_reparameterization():
    vae = VariationalAutoencoder(input_dim=4, hidden_dim=128, latent_dim=2)
    mu = torch.zeros(5, 4)
    logvar = torch.zeros(5, 4)
    z = vae.reparameterize(mu, logvar)
    assert z.shape == (5, 4)

# VAE-05: reparameterize edge case - mismatched shapes
def test_reparameterization_edge():
    vae = VariationalAutoencoder(input_dim=4, hidden_dim=128, latent_dim=2)
    mu = torch.zeros(5, 3)
    logvar = torch.zeros(5, 2)
    with pytest.raises(RuntimeError):
        vae.reparameterize(mu, logvar)

# VAE-06: decode correct shape
def test_decoder():
    vae = VariationalAutoencoder(input_dim=4, hidden_dim=128, latent_dim=3)
    z = torch.randn(5, 3)
    assert vae.decode(z).shape == (5, 4)

# VAE-07: decode edge case - wrong latent dim
def test_decoder_edge():
    vae = VariationalAutoencoder(input_dim=4, hidden_dim=128, latent_dim=3)
    z = torch.randn(5, 2)
    with pytest.raises(RuntimeError):
        vae.decode(z)

# VAE-08: forward correct shapes
def test_forward():
    vae = VariationalAutoencoder(input_dim=4, hidden_dim=128, latent_dim=3)
    x = torch.randn(5, 4)
    recon, mu, logvar = vae.forward(x)
    assert recon.shape == (5, 4)
    assert mu.shape == (5, 3)
    assert logvar.shape == (5, 3)

# VAE-09: forward edge case - wrong input dim
def test_forward_edge():
    vae = VariationalAutoencoder(input_dim=4, hidden_dim=128, latent_dim=3)
    x = torch.randn(5, 5)
    with pytest.raises(RuntimeError):
        vae.forward(x)

# VAE-10: forward no NaN or inf
def test_forward_no_nan():
    vae = VariationalAutoencoder(input_dim=4, hidden_dim=128, latent_dim=3)
    x = torch.randn(8, 4)
    recon, mu, logvar = vae.forward(x)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()