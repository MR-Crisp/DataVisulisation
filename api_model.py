from fastapi import FastAPI, UploadFile, File
import io
import json

from voronoi_algorithm import voronoi_finite_polygons,plot_voronoi
import umap
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import plotly.express as px
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi.responses import JSONResponse

#from my files
from main import train_vae,get_tensor
from VAE import VariationalAutoencoder
from GMM_bic import GMM
from Dataset import StaticDataset


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # your React port
    allow_methods=["*"],
    allow_headers=["*"],
)


#state (global)
state = {
    "D": None,
    "X_tensor": None,
    "sample_size": None,
    "vae_model": None,
    "latent_vectors": None,
    "labels": None,
}

@app.get("/")
def root():
    return {"message": "Welcome to the API!"}

@app.post("/Upload_CSV")
async def upload_csv(file: UploadFile = File(...),target_col: str = "Cover_Type"):
    contents = await file.read()
    csv = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
    D = StaticDataset(target_col=target_col)######NNNEEEEEDDDDSSS to be changed
    D.input_dataset(csv)
    D.preprocess()
    X_tensor = get_tensor(D.df)
    sample_size = int(0.1 * len(X_tensor))  # Use 10% of the data for training
    state["D"] = D
    state["X_tensor"] = X_tensor
    state["sample_size"] = sample_size


@app.post("/vae_training")
def vae_training():
    D = state["D"]
    X_tensor = state["X_tensor"]
    input_dim = D.df.shape[1] - 1 if "Cover_Type" in D.df.columns else D.df.shape[1]
    dataset = TensorDataset(X_tensor)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    vae_model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=128, latent_dim=3)
    train_vae(vae_model, train_loader, epochs=20, lr=0.001)
    with torch.no_grad():
        mu, _ = vae_model.encode(X_tensor)
        state["latent_vectors"] = mu.numpy()

    state["vae_model"] = vae_model

@app.post("/GMM_bic")#needs to be updated
def gmm_bic():
    latent_vectors = state["latent_vectors"]
    gmm_model = GMM()
    labels, gmm = gmm_model.GMM_calc(latent_vectors)
    state["labels"] = labels

    fig = gmm_model.visual(latent_vectors, labels, gmm)  # figure to return
    return JSONResponse(content=json.loads(fig.to_json()))#turning figure to json for front end

@app.get("/voronoi")
def voronoi(lanent_space):###########need to make this plotly compatible
    latent_vectors = state["latent_vectors"]
    sample_size = state["sample_size"]
    D = state["D"]
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

    return JSONResponse(content=json.loads(fig.to_json()))

@app.get("/heatmap")
def heatmap(latent):
    pass

@app.get("/particle")
def particle(latent):
    pass
