from fastapi import FastAPI
import pandas as pd
import io
from fastapi import UploadFile, File
from voronoi_algorithm import voronoi_finite_polygons,plot_voronoi
import umap
from main import train_vae,get_tensor, StaticDataset
import torch
from GMM_bic import GMM

app = FastAPI()
df =  None # main df for the dataset
latent_vectors = None #from VAE and dataset
@app.get("/")
def root():
    return {"message": "Welcome to the API!"}

@app.post("Upload_CSV")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

@app.post("/data_cleaning")
def data_cleaning(df: pd.DataFrame):
    pass

@app.post("/vae_training")
def vae_training(D: StaticDataset):
    input_dim = D.df.shape[1] - 1 if 'Cover_Type' in D.df.columns else D.df.shape[1]
    X_tensor = get_tensor(D.df)
    sample_size = int(0.1 * len(X_tensor))  # Use 10% of the data for training
    X_tensor = X_tensor[:sample_size]  # Take the first 10% of the data for training
    print("Training new VAE model...")
    dataset = TensorDataset(X_tensor)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    vae_model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=128, latent_dim=3)
    train_vae(vae_model, train_loader, epochs=60, lr=0.001)
    save_model(vae_model, 'vae_model.pth')
    print("VAE model trained and saved as 'vae_model.pth'.")
    with torch.no_grad():
        mu, logvar = vae_model.encode(X_tensor)
        latent_vectors = mu.numpy()


@app.post("GMM_bic")#needs to be updated
def gmm_bic(latent_vectors: torch.Tensor):
    gmm_model = GMM()
    labels, gmm = gmm_model.GMM_calc(latent_vectors)
    print(f"Number of clusters found: {len(np.unique(labels))}")
    print(f"GMM converged: {gmm.converged_}")
    print(f"Cluster distribution: {np.bincount(labels)}")

    gmm_model.visual(latent_vectors, labels, gmm)

@app.get("/voronoi")
def voronoi(lanent_space):
    pass

@app.get("/heatmap")
def heatmap(latent):
    pass

@app.get("particle")
def particle(latent):
    pass