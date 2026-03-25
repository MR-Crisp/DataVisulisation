import numpy as np
import pandas as pd
import torch
import os
import base64
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import kagglehub
from kagglehub import KaggleDatasetAdapter

from VAE import VariationalAutoencoder
from GMM_bic import GMM
from main import StaticDataset, get_tensor, load_model, train_vae, save_model  # adjust as needed

