from fastapi import FastAPI, UploadFile, File
import io
import json
import os
from voronoi_algorithm import voronoi_finite_polygons,plot_voronoi
import umap
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import plotly.express as px
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi.responses import JSONResponse
import plotly.graph_objects as go

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
    "target_col":None,
    "feature_names":[],
    "label_encoder":None
}

@app.get("/")
def root():
    return {"message": "Welcome to the API!"}

@app.post("/Upload_CSV")
async def upload_csv(file: UploadFile = File(...), target_col: str = "Cover_Type"):
    contents = await file.read()
    csv = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
    D = StaticDataset(target_col=target_col)
    D.input_dataset(csv)
    D.preprocess()

    X_tensor = get_tensor(D.df, target_col=target_col)  # ✅ pass target_col
    sample_size = int(0.1 * len(X_tensor))

    state["D"] = D
    state["X_tensor"] = X_tensor[:sample_size]  # ✅ slice here, not later
    state["sample_size"] = sample_size
    state["feature_names"] = [col for col in D.df.columns if col != target_col]
    state["target_col"] = target_col
    

@app.post("/vae_training")
def vae_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = state["D"]
    X_tensor = state["X_tensor"]
    target_col = state["target_col"]

    input_dim = D.df.shape[1] - 1 if (target_col and target_col in D.df.columns) else D.df.shape[1]
    model_path = "vae_model.pth"
    meta_path = "vae_model_meta.npy"

    vae_model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=128, latent_dim=3)

    if os.path.exists(model_path) and os.path.exists(meta_path):
        saved_input_dim = int(np.load(meta_path))
        if saved_input_dim == input_dim:
            print(f"Loading saved VAE model (input_dim={input_dim})...")
            vae_model.load_state_dict(torch.load(model_path, map_location=device))
            vae_model.eval()
        else:
            print(f"input_dim mismatch ({saved_input_dim} vs {input_dim}) — retraining...")
            for f in ["vae_model.pth", "vae_model_meta.npy", "gmm_model.pkl", "gmm_labels.npy", "umap_coords.npy"]:
                if os.path.exists(f): os.remove(f)
            dataset = TensorDataset(X_tensor)
            train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
            train_vae(vae_model, train_loader, epochs=20, lr=0.001)
            torch.save(vae_model.state_dict(), model_path)
            np.save(meta_path, np.array(input_dim))
    else:
        print("Training new VAE model...")
        dataset = TensorDataset(X_tensor)
        train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
        train_vae(vae_model, train_loader, epochs=20, lr=0.001)
        torch.save(vae_model.state_dict(), model_path)
        np.save(meta_path, np.array(input_dim))
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
        target_col = state["target_col"]
        labels_path = "gmm_labels.npy"

        if state["labels"] is None and os.path.exists(labels_path):
            state["labels"] = np.load(labels_path)

        # ✅ Use target col if present, else fall back to GMM labels
        if target_col and target_col in D.df.columns:
            all_labels = D.df[target_col].values[:sample_size]
            # ✅ Convert to integers if possible, else encode to ints
            try:
                all_labels = all_labels.astype(int)
            except (ValueError, TypeError):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                all_labels = le.fit_transform(all_labels)
                state["label_encoder"] = le  # save so we can decode later
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

        unique_classes = np.unique(cover_labels)
        palette = px.colors.qualitative.Bold
        class_colour = {cls: palette[i % len(palette)] for i, cls in enumerate(unique_classes)}

        # ✅ Generic class names — just use the label value itself
        class_names = {cls: str(cls) for cls in unique_classes}

        fig = plot_voronoi(coords_2d, cover_labels, class_colour, class_names)
        return JSONResponse(content=json.loads(fig.to_json()))

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/heatmap")
def heatmap(z1: float, z2: float, z3: float):
    try:
        import base64, io, matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        vae_model = state["vae_model"]
        if vae_model is None:
            return JSONResponse(status_code=400, content={"error": "Train VAE first"})

        device = next(vae_model.parameters()).device
        z = torch.tensor([[z1, z2, z3]], dtype=torch.float32).to(device)

        vae_model.eval()  # ✅ BatchNorm needs eval mode for batch size of 1
        with torch.no_grad():
            features = vae_model.decode(z).cpu().numpy().flatten()

        feature_names = state["feature_names"]
        n_features = len(features)

        fig, ax = plt.subplots(figsize=(16, 4))
        reshaped = features.reshape(1, -1)
        im = ax.imshow(reshaped, cmap='RdYlBu_r', aspect='auto', interpolation='bilinear')
        ax.set_xticks(np.arange(n_features))
        ax.set_xticklabels([name[:10] for name in feature_names], rotation=90, fontsize=8)
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='Feature Value', shrink=0.8)
        ax.set_title(f'Generated Sample — Feature Heatmap ({n_features} features)')
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return JSONResponse(content={"image": f"data:image/png;base64,{encoded}"})

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/particle")
def particle(z1: float, z2: float, z3: float):
    try:
        vae_model = state["vae_model"]
        if vae_model is None:
            return JSONResponse(status_code=400, content={"error": "Train VAE first"})

        device = next(vae_model.parameters()).device
        z = torch.tensor([[z1, z2, z3]], dtype=torch.float32).to(device)

        vae_model.eval()  # ✅ BatchNorm needs eval mode for batch size of 1
        with torch.no_grad():
            features = vae_model.decode(z).cpu().numpy().flatten()

        feature_names = state["feature_names"]
        n = len(features)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        norm_vals = (features - features.min()) / (features.max() - features.min() + 1e-8)
        radii = 1 + norm_vals * 0.8
        x = (radii * np.cos(angles)).tolist()
        y = (radii * np.sin(angles)).tolist()
        sizes = (10 + norm_vals * 30).tolist()

        hover_texts = [f"{feature_names[i]}: {features[i]:.3f}" for i in range(n)]

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=sizes,
                color=norm_vals.tolist(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Feature value')
            ),
            text=hover_texts,
            hoverinfo='text'
        ))

        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta).tolist(),
            y=np.sin(theta).tolist(),
            mode='lines',
            line=dict(color='gray', width=2, dash='dash'),
            showlegend=False
        ))

        fig.update_layout(
            title='Particle System — Each particle is a feature',
            xaxis=dict(visible=False, range=[-2.2, 2.2]),
            yaxis=dict(visible=False, range=[-2.2, 2.2]),
            showlegend=False
        )

        return JSONResponse(content=json.loads(fig.to_json()))

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})