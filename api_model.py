from fastapi import FastAPI
import pandas as pd
import io
from fastapi import UploadFile, File
from voronoi_algorithm import voronoi_finite_polygons,plot_voronoi
import umap

app = FastAPI()
df =  None # main df for the dataset
latent_space = None #from VAE and dataset
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

@app.post("/vae_GMM")
def vae(df: pd.DataFrame):
    pass

@app.get("/voronoi")
def voronoi(lanent_space):
    pass

@app.get("/heatmap")
def heatmap(latent):
    pass

@app.get("particle")
def particle(latent):
    pass