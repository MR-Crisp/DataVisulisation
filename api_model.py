from fastapi import FastAPI
import io
from fastapi import UploadFile, File
from voronoi_algorithm import voronoi_finite_polygons,plot_voronoi
import umap
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import plotly.express as px

#from my files
from main import train_vae,get_tensor, StaticDataset
from VAE import VariationalAutoencoder
from GMM_bic import GMM


app = FastAPI()
D =  None # main df for the dataset
latent_vectors = None #from VAE and dataset
@app.get("/")
def root():
    return {"message": "Welcome to the API!"}

@app.post("Upload_CSV")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    csv = io.BytesIO(contents)
    D = StaticDataset()
    D.input_covertype_dataset(csv)
    D.clean_covertype_dataset()
    D.normalise_covertype_data()
    input_dim = D.df.shape[1] - 1 if 'Cover_Type' in D.df.columns else D.df.shape[1]
    X_tensor = get_tensor(D.df)
    sample_size = int(0.1 * len(X_tensor))  # Use 10% of the data for training
    X_tensor = X_tensor[:sample_size]  # Take the first 10% of the data for training

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
def voronoi(lanent_space):###########need to make this plotly compatible
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,  # local neighbourhood size — increase for smoother layout
        min_dist=0.1,  # how tightly points cluster — 0.0 = tightest
        random_state=42,
        metric='euclidean'
    )
    coords_2d = reducer.fit_transform(latent_vectors)  # shape (N, 2)

    # Pull Cover_Type labels aligned to the sample
    cover_labels = D.df['Cover_Type'].values[:sample_size].astype(int)
    unique_classes = np.unique(cover_labels)
    n_classes = len(unique_classes)

    # Build a colour map
    palette = px.colors.qualitative.Bold
    class_colour = {cls: palette[i % len(palette)] for i, cls in enumerate(unique_classes)}
    cover_type_names = {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas-fir",
        7: "Krummholz"}

    fig = plot_voronoi(coords_2d, cover_labels, class_colour, cover_type_names)

    fig.show()

@app.get("/heatmap")
def heatmap(latent):
    pass

@app.get("particle")
def particle(latent):
    pass