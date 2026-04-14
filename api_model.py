from fastapi import FastAPI, UploadFile, File, Request
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
from main import train_vae,get_tensor,get_vae_config
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

KNOWN_LABEL_MAPS = {
    "Cover_Type": {
        1: "Spruce/Fir",
        2: "Lodgepole Pine",
        3: "Ponderosa Pine",
        4: "Cottonwood/Willow",
        5: "Aspen",
        6: "Douglas Fir",
        7: "Krummholz"
    }
}


def predict_class(features: np.ndarray) -> dict:
    D = state["D"]
    target_col = state["target_col"]
    X_array = state["X_tensor"].numpy()
    all_labels = D.df[target_col].values[:len(X_array)]

    label_map = KNOWN_LABEL_MAPS.get(target_col, state.get("label_map", {}))

    def resolve(val):
        try:
            return label_map.get(int(val), str(val))
        except (ValueError, TypeError):
            return str(val)

    unique_classes = np.unique(all_labels)

    # Build balanced pool — take up to 20 samples per class
    per_class = 20
    pool_idx = []
    for cls in unique_classes:
        cls_idx = np.where(all_labels == cls)[0]
        chosen = cls_idx[:per_class] if len(cls_idx) <= per_class else np.random.choice(cls_idx, per_class, replace=False)
        pool_idx.extend(chosen.tolist())

    pool_idx = np.array(pool_idx)
    pool_X = X_array[pool_idx]
    pool_labels = all_labels[pool_idx]

    # Find k nearest in the balanced pool
    dists = np.linalg.norm(pool_X - features, axis=1)
    k = min(20, len(pool_X))
    top_k_idx = np.argsort(dists)[:k]
    top_k_labels = pool_labels[top_k_idx]

    unique, counts = np.unique(top_k_labels, return_counts=True)
    class_distribution = {
        resolve(cls): float(cnt / k)
        for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])
    }
    predicted_class = max(class_distribution, key=class_distribution.get)

    return {
        "predicted_class": predicted_class,
        "confidence": float(class_distribution[predicted_class]),
        "class_distribution": class_distribution
    }

@app.get("/")
def root():
    return {"message": "Welcome to the API!"}

@app.post("/Upload_CSV")
async def upload_csv(file: UploadFile = File(...), target_col: str = "Cover_Type", method: str = "vae"):
    import scipy.sparse as sp
    import hashlib

    contents = await file.read()
    csv = pd.read_csv(io.BytesIO(contents), encoding='latin-1')

    D = StaticDataset(target_col=target_col)
    D.input_dataset(csv)
    D.preprocess()

    vae_config = get_vae_config(D)

    # Fingerprint based on shape + column names + target col
    # This uniquely identifies the dataset so the cache is invalidated on new uploads
    fingerprint_str = f"{csv.shape}_{list(csv.columns)}_{target_col}"
    fingerprint = hashlib.md5(fingerprint_str.encode()).hexdigest()[:8]
    vae_config["dataset_fingerprint"] = fingerprint

    state["vae_config"] = vae_config

    X_array = D.X.toarray() if sp.issparse(D.X) else D.X
    X_tensor = torch.tensor(X_array.astype('float32'))
    sample_size = max(200, int(0.1 * len(X_tensor)))
    sample_size = min(sample_size, len(X_tensor))

    state["D"] = D
    state["X_tensor"] = X_tensor[:sample_size]
    state["sample_size"] = sample_size
    state["feature_names"] = [col for col in D.df.columns if col != target_col]
    state["target_col"] = target_col
    state["method"] = method
    state["labels"] = None
    state["latent_vectors"] = None
    state["vae_model"] = None

    print(f"Dataset fingerprint: {fingerprint}")
    print(f"Dataset types: {D.types}")

@app.post("/vae_training")
def vae_training():
    import json
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = state["X_tensor"]
    cfg = state["vae_config"]
    model_path = "vae_model.pth"
    config_path = "vae_config.json"

    vae_model = VariationalAutoencoder(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        beta=cfg["beta"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
    )

    if os.path.exists(model_path) and os.path.exists(config_path):
        with open(config_path) as f:
            saved_cfg = json.load(f)

        if saved_cfg == cfg:
            print("Loading saved VAE model...")
            vae_model.load_state_dict(torch.load(model_path, map_location=device))
            vae_model.eval()
        else:
            print(f"Config mismatch — retraining. Old: {saved_cfg}, New: {cfg}")
            for fname in ["vae_model.pth", "vae_config.json", "gmm_model.pkl", "gmm_labels.npy", "umap_coords.npy"]:
                if os.path.exists(fname): os.remove(fname)
            dataset = TensorDataset(X_tensor)
            train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
            train_vae(vae_model, train_loader, epochs=120, lr=0.001, beta_target=cfg["beta"])  # ✅ pass beta
            torch.save(vae_model.state_dict(), model_path)
            with open(config_path, "w") as f:
                json.dump(cfg, f)
    else:
        print("Training new VAE model...")
        dataset = TensorDataset(X_tensor)
        train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
        train_vae(vae_model, train_loader, epochs=120, lr=0.001, beta_target=cfg["beta"])  # ✅ pass beta
        torch.save(vae_model.state_dict(), model_path)
        with open(config_path, "w") as f:
            json.dump(cfg, f)
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

    # Build the plot manually so we can set customdata on the points
    probs = gmm.predict_proba(latent_vectors)
    n_clusters = probs.shape[1]

    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('tab10')
    cluster_colors = cmap(np.arange(n_clusters))[:, :3]
    point_colors = probs @ cluster_colors
    point_colors_hex = [
        f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})'
        for c in point_colors
    ]

    fig = go.Figure()

    # Main scatter — customdata carries the full latent vector for click handling
    fig.add_trace(go.Scatter3d(
        x=latent_vectors[:, 0].tolist(),
        y=latent_vectors[:, 1].tolist(),
        z=latent_vectors[:, 2].tolist(),
        mode='markers',
        marker=dict(size=4, color=point_colors_hex, opacity=0.8),
        customdata=latent_vectors.tolist(),  # full latent vector sent back on click
        hovertemplate=(
            "Z1: %{x:.3f}<br>"
            "Z2: %{y:.3f}<br>"
            "Z3: %{z:.3f}<br>"
            "<extra></extra>"
        ),
        name='Data points'
    ))

    # Centroids
    fig.add_trace(go.Scatter3d(
        x=gmm.means_[:, 0].tolist(),
        y=gmm.means_[:, 1].tolist(),
        z=gmm.means_[:, 2].tolist(),
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='black')),
        name='Centroids',
        hovertemplate="Centroid<br>Z1: %{x:.3f}<br>Z2: %{y:.3f}<br>Z3: %{z:.3f}<extra></extra>"
    ))

    fig.update_layout(
        title='GMM Latent Space — click any point to explore it',
        scene=dict(
            xaxis_title='Z1',
            yaxis_title='Z2',
            zaxis_title='Z3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        ),
        width=800, height=600
    )

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

        cfg = state["vae_config"]
        latent_dim = cfg["latent_dim"]
        device = next(vae_model.parameters()).device

        z_np = np.zeros(latent_dim, dtype=np.float32)
        z_np[0] = z1
        if latent_dim > 1: z_np[1] = z2
        if latent_dim > 2: z_np[2] = z3
        z = torch.tensor([z_np], dtype=torch.float32).to(device)

        vae_model.eval()
        with torch.no_grad():
            features = vae_model.decode(z).cpu().numpy().flatten()

        class_info = predict_class(features)
        feature_names = state["feature_names"]
        n_features = len(features)

        fig, ax = plt.subplots(figsize=(16, 4))
        reshaped = features.reshape(1, -1)
        im = ax.imshow(reshaped, cmap='RdYlBu_r', aspect='auto', interpolation='bilinear')
        ax.set_xticks(np.arange(n_features))
        ax.set_xticklabels([name[:10] for name in feature_names], rotation=90, fontsize=8)
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, label='Feature Value', shrink=0.8)
        ax.set_title(
            f'Predicted: {class_info["predicted_class"]}  '
            f'(confidence: {class_info["confidence"] * 100:.0f}%)',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return JSONResponse(content={
            "image": f"data:image/png;base64,{encoded}",
            "class_info": class_info
        })

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

        cfg = state["vae_config"]
        latent_dim = cfg["latent_dim"]
        device = next(vae_model.parameters()).device

        z_np = np.zeros(latent_dim, dtype=np.float32)
        z_np[0] = z1
        if latent_dim > 1: z_np[1] = z2
        if latent_dim > 2: z_np[2] = z3
        z = torch.tensor([z_np], dtype=torch.float32).to(device)

        vae_model.eval()
        with torch.no_grad():
            features = vae_model.decode(z).cpu().numpy().flatten()

        class_info = predict_class(features)
        feature_names = state["feature_names"]
        n = len(features)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        norm_vals = (features - features.min()) / (features.max() - features.min() + 1e-8)
        radii = 1 + norm_vals * 0.8
        x = (radii * np.cos(angles)).tolist()
        y = (radii * np.sin(angles)).tolist()
        sizes = (10 + norm_vals * 30).tolist()
        hover_texts = [f"{feature_names[i]}: {features[i]:.3f}" for i in range(n)]

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
            title=f'Predicted: {class_info["predicted_class"]} ({class_info["confidence"] * 100:.0f}% confidence)',
            xaxis=dict(visible=False, range=[-2.2, 2.2]),
            yaxis=dict(visible=False, range=[-2.2, 2.2]),
            showlegend=False
        )

        response = json.loads(fig.to_json())
        response["class_info"] = class_info
        return JSONResponse(content=response)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/config")
def get_config():
    cfg = state.get("vae_config")
    latent = state.get("latent_vectors")
    if cfg is None:
        return JSONResponse(status_code=400, content={"error": "No model trained yet"})

    response = {"latent_dim": cfg["latent_dim"]}

    # Return actual min/max per latent dimension from the encoded data
    if latent is not None:
        response["latent_ranges"] = [
            {"min": float(latent[:, i].min()), "max": float(latent[:, i].max())}
            for i in range(latent.shape[1])
        ]
    else:
        response["latent_ranges"] = [
            {"min": -5.0, "max": 5.0} for _ in range(cfg["latent_dim"])
        ]

    return JSONResponse(content=response)

@app.post("/label_map")
async def set_label_map(request: Request):
    body = await request.json()  # expects {"1": "Spruce/Fir", "2": "Lodgepole Pine", ...}
    state["label_map"] = {int(k): v for k, v in body.items()}
    return {"message": f"Label map set with {len(state['label_map'])} entries"}
