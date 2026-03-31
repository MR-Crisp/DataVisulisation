from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import plotly.graph_objects as go
from main import main

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/graph")
def get_graph():
    fig = main()
    return fig.to_json()
