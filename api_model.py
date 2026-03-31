from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import plotly.graph_objects as go

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the API!"}
