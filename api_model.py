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
    allow_origins=["http://localhost:3000"],
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


import os

@app.post("/vae_training")
def vae_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = state["D"]
    X_tensor = state["X_tensor"]
    input_dim = D.df.shape[1] - 1 if "Cover_Type" in D.df.columns else D.df.shape[1]
    model_path = "vae_model.pth"

    vae_model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=128, latent_dim=3)

    if os.path.exists(model_path):
        print("Loading saved VAE model...")
        vae_model.load_state_dict(torch.load(model_path, map_location=device))
        vae_model.eval()
    else:
        print("Training new VAE model...")
        dataset = TensorDataset(X_tensor)
        train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
        train_vae(vae_model, train_loader, epochs=20, lr=0.001)
        torch.save(vae_model.state_dict(), model_path)
        print("Model saved.")

    with torch.no_grad():
        X_gpu = X_tensor.to(device)
        vae_model = vae_model.to(device)
        mu, _ = vae_model.encode(X_gpu)
        state["latent_vectors"] = mu.cpu().numpy()

    state["vae_model"] = vae_model

@app.post("/GMM_bic")
def gmm_bic():
    import pickle
    latent_vectors = state["latent_vectors"]
    gmm_path = "gmm_model.pkl"
    labels_path = "gmm_labels.npy"

    if os.path.exists(gmm_path) and os.path.exists(labels_path):
        print("Loading saved GMM model...")
        with open(gmm_path, "rb") as f:
            gmm = pickle.load(f)
        labels = np.load(labels_path)
    else:
        print("Running GMM clustering...")
        gmm_model = GMM()
        labels, gmm = gmm_model.GMM_calc(latent_vectors)
        with open(gmm_path, "wb") as f:
            pickle.dump(gmm, f)
        np.save(labels_path, labels)
        print("GMM saved.")

    state["labels"] = labels

    gmm_visual = GMM()
    fig = gmm_visual.visual(latent_vectors, labels, gmm)
    return JSONResponse(content=json.loads(fig.to_json()))
@app.get("/voronoi")
def voronoi():
    try:
        latent_vectors = state["latent_vectors"]
        D = state["D"]
        sample_size = state["sample_size"]
        labels_path = "gmm_labels.npy"

        if state["labels"] is None and os.path.exists(labels_path):
            state["labels"] = np.load(labels_path)

        if "Cover_Type" in D.df.columns:
            all_labels = D.df["Cover_Type"].values[:sample_size].astype(int)
        else:
            all_labels = state["labels"]

        max_points = 5000
        n = min(len(latent_vectors), len(all_labels))
        if n > max_points:
            idx = np.random.choice(n, max_points, replace=False)
        else:
            idx = np.arange(n)

        latent_sample = latent_vectors[idx]
        cover_labels = all_labels[idx]
        print(f"Sampled {len(idx)} points, latent: {latent_sample.shape}, labels: {cover_labels.shape}")

        umap_path = "umap_coords.npy"
        if os.path.exists(umap_path):
            coords_2d = np.load(umap_path)
            print(f"Loaded UMAP coords from cache: {coords_2d.shape}")
        else:
            print("Running UMAP...")
            reducer = umap.UMAP(n_components=2, n_neighbors=15,
                                min_dist=0.1, random_state=42, metric="euclidean")
            coords_2d = reducer.fit_transform(latent_sample)
            np.save(umap_path, coords_2d)
            print(f"UMAP done: {coords_2d.shape}")

        print(f"coords_2d: {coords_2d.shape}, cover_labels: {cover_labels.shape}")

        unique_classes = np.unique(cover_labels)
        palette = px.colors.qualitative.Bold
        class_colour = {cls: palette[i % len(palette)] for i, cls in enumerate(unique_classes)}
        cover_type_names = {
            1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
            4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz",
        }

        print("Running plot_voronoi...")
        fig = plot_voronoi(coords_2d, cover_labels, class_colour, cover_type_names)
        print("Serializing...")
        return JSONResponse(content=json.loads(fig.to_json()))

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"VORONOI ERROR:\n{error_msg}")
        return JSONResponse(status_code=500, content={"error": str(e), "detail": error_msg})

@app.get("/heatmap")
def heatmap(latent):
    pass

@app.get("/particle")
def particle(latent):
    pass
