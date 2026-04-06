import pytest
import torch
from VAE import VariationalAutoencoder
from GMM_bic import GMM
from unittest.mock import patch
from sklearn.mixture import GaussianMixture
import numpy as np

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

# GMM-01: init
def test_GMM_init():
    gmm = GMM()
    assert isinstance(gmm, GMM)

# GMM-02: calculate_bic_for_gmm returns correct number of values
def test_calculate_bic_for_gmm():
    gmm = GMM()
    X = np.random.rand(10, 3)
    bic_values = gmm.calculate_bic_for_gmm(X, max_clusters=5)
    assert len(bic_values) == 5

# GMM-03: calculate_bic_for_gmm with max_clusters=1
def test_calculate_bic_for_gmm_single_cluster():
    gmm = GMM()
    X = np.random.rand(10, 3)
    bic_values = gmm.calculate_bic_for_gmm(X, max_clusters=1)
    assert len(bic_values) == 1

# GMM-04: calculate_bic_for_gmm edge case - invalid data shape (1D array)
def test_calculate_bic_for_gmm_invalid_shape():
    gmm = GMM()
    X = np.random.rand(10)
    with pytest.raises(Exception):
        gmm.calculate_bic_for_gmm(X, max_clusters=3)

# GMM-05: GMM_calc returns labels and trained gmm
def test_GMM_calc():
    gmm = GMM()
    X = np.random.rand(20, 3)
    labels, fitted_gmm = gmm.GMM_calc(X)
    assert len(labels) == 20
    assert isinstance(fitted_gmm, GaussianMixture)

# GMM-06: GMM_calc edge case - very small dataset
def test_GMM_calc_small_dataset():
    gmm = GMM()
    X = np.random.rand(2, 3)
    labels, fitted_gmm = gmm.GMM_calc(X)
    assert len(labels) == 2

# GMM-07: GMM_calc edge case - empty dataset
def test_GMM_calc_empty():
    gmm = GMM()
    X = np.array([])
    with pytest.raises(Exception):
        gmm.GMM_calc(X)

# GMM-08: visual runs without errors
def test_visual():
    gmm_instance = GMM()
    X = np.random.rand(10, 3)
    labels, fitted_gmm = gmm_instance.GMM_calc(X)
    with patch('plotly.graph_objects.Figure.show'):
        gmm_instance.visual(X, labels, fitted_gmm)

# GMM-09: visual edge case - mismatched labels length
def test_visual_mismatched_labels():
    gmm_instance = GMM()
    X = np.random.rand(10, 3)
    labels = np.zeros(5)
    _, fitted_gmm = gmm_instance.GMM_calc(np.random.rand(10, 3))
    with pytest.raises(Exception):
        gmm_instance.visual(X, labels, fitted_gmm)

# GMM-10: visual edge case - insufficient dimensions (2D instead of 3D)
def test_visual_insufficient_dimensions():
    gmm_instance = GMM()
    X = np.random.rand(10, 2)
    labels = np.zeros(10)
    _, fitted_gmm = gmm_instance.GMM_calc(np.random.rand(10, 3))
    with pytest.raises((IndexError, Exception)):
        gmm_instance.visual(X, labels, fitted_gmm)



