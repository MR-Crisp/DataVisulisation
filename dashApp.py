import numpy as np
import pandas as pd
import torch
import os
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for safer threading with Dash
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
import kagglehub
from kagglehub import KaggleDatasetAdapter

from VAE import VariationalAutoencoder
from GMM_bic import GMM
from main import StaticDataset, get_tensor, load_model, train_vae, save_model, load_model

FEATURE_DEFINITIONS = {
    # Continuous features
    'Elevation': 0,
    'Aspect': 1,
    'Slope': 2,
    'Horizontal_Distance_To_Hydrology': 3,
    'Vertical_Distance_To_Hydrology': 4,
    'Horizontal_Distance_To_Roadways': 5,
    'Hillshade_9am': 6,
    'Hillshade_Noon': 7,
    'Hillshade_3pm': 8,
    
    # Wilderness area binary features
    'Wilderness_Area_Rawah': 9,
    'Wilderness_Area_Neota': 10,
    'Wilderness_Area_Comanche': 11,
    'Wilderness_Area_Cache_La_Poudre': 12,
    
    # Soil type binary features (40 soil types)
    'Soil_Type_1': 13, 'Soil_Type_2': 14, 'Soil_Type_3': 15, 'Soil_Type_4': 16,
    'Soil_Type_5': 17, 'Soil_Type_6': 18, 'Soil_Type_7': 19, 'Soil_Type_8': 20,
    'Soil_Type_9': 21, 'Soil_Type_10': 22, 'Soil_Type_11': 23, 'Soil_Type_12': 24,
    'Soil_Type_13': 25, 'Soil_Type_14': 26, 'Soil_Type_15': 27, 'Soil_Type_16': 28,
    'Soil_Type_17': 29, 'Soil_Type_18': 30, 'Soil_Type_19': 31, 'Soil_Type_20': 32,
    'Soil_Type_21': 33, 'Soil_Type_22': 34, 'Soil_Type_23': 35, 'Soil_Type_24': 36,
    'Soil_Type_25': 37, 'Soil_Type_26': 38, 'Soil_Type_27': 39, 'Soil_Type_28': 40,
    'Soil_Type_29': 41, 'Soil_Type_30': 42, 'Soil_Type_31': 43, 'Soil_Type_32': 44,
    'Soil_Type_33': 45, 'Soil_Type_34': 46, 'Soil_Type_35': 47, 'Soil_Type_36': 48,
    'Soil_Type_37': 49, 'Soil_Type_38': 50, 'Soil_Type_39': 51, 'Soil_Type_40': 52,
}

# Create feature names list (52 features)
feature_names = [name for name, idx in sorted(FEATURE_DEFINITIONS.items(), key=lambda x: x[1])]

# Group features by category for better organisation
FEATURE_GROUPS = {
    'Topographic': ['Elevation', 'Aspect', 'Slope'],
    'Hydrology': ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'],
    'Infrastructure': ['Horizontal_Distance_To_Roadways'],
    'Hillshade': ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'],
    'Wilderness Areas': [f'Wilderness_Area_{area}' for area in ['Rawah', 'Neota', 'Comanche', 'Cache_La_Poudre']],
    'Soil Types': [f'Soil_Type_{i}' for i in range(1, 41)]
}


file_path = "./covertype.csv"
testset = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS, "zsinghrahulk/covertype-forest-cover-types", file_path)

D = StaticDataset()
D.input_covertype_dataset(testset)
D.clean_covertype_dataset()
D.normalise_covertype_data()
print(f"Dataset shape: {D.df.shape}")

input_dim = D.df.shape[1] - 1 if 'Cover_Type' in D.df.columns else D.df.shape[1]
X_tensor = get_tensor(D.df)
sample_size = int(0.1 * len(X_tensor))
X_tensor = X_tensor[:sample_size]

# Load or train VAE
if os.path.exists('vae_model.pth'):
    print("Loading existing VAE model...")
    vae_model = load_model('vae_model.pth', input_dim=input_dim, hidden_dim=128, latent_dim=3)
else:
    print("Training new VAE model...")
    dataset = TensorDataset(X_tensor)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    vae_model = VariationalAutoencoder(input_dim=input_dim, hidden_dim=128, latent_dim=3)
    train_vae(vae_model, train_loader, epochs=60, lr=0.001)
    save_model(vae_model, 'vae_model.pth')

# Encode data to latent space
with torch.no_grad():
    mu, logvar = vae_model.encode(X_tensor)
    latent_vectors = mu.numpy()

# Fit GMM (with BIC)
gmm_model = GMM()
labels, gmm = gmm_model.GMM_calc(latent_vectors)
print(f"Number of clusters: {len(np.unique(labels))}")

# Feature names (excluding target column)
feature_names = [col for col in D.df.columns if col != 'Cover_Type']
n_features = len(feature_names)

def decode_latent(z):
    with torch.no_grad():
        z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
        reconstructed = vae_model.decode(z_tensor).numpy().flatten()
    return reconstructed

def create_feature_table(features):
    """Create a detailed table of feature values with categories"""
    table_data = []
    
    for group_name, group_features in FEATURE_GROUPS.items():
        # Add group header
        table_data.append({
            'Category': group_name,
            'Feature Name': '━━━━━━━━━━━━━━━━━━━━',
            'Value': '',
            'Interpretation': ''
        })
        
        for feature in group_features:
            # Find the index of this feature
            idx = FEATURE_DEFINITIONS.get(feature)
            if idx is not None and idx < len(features):
                value = features[idx]
                
                # Generate interpretation based on feature type
                if 'Soil Type' in feature:
                    interpretation = ' Present' if value > 0.5 else 'Absent'
                elif 'Wilderness Area:' in feature:
                    interpretation = ' In this area' if value > 0.5 else 'Not in this area'
                elif 'Elevation' in feature:
                    if value > 0.6:
                        interpretation = 'High elevation'
                    elif value < 0.3:
                        interpretation = 'Low elevation'
                    else:
                        interpretation = 'Moderate elevation'
                elif 'Slope' in feature:
                    if value > 0.6:
                        interpretation = 'Steep'
                    elif value < 0.3:
                        interpretation = 'Flat'
                    else:
                        interpretation = 'Moderate slope'
                elif 'Hydrology' in feature:
                    if value < 0.3:
                        interpretation = 'Close to water'
                    elif value > 0.6:
                        interpretation = 'Far from water'
                    else:
                        interpretation = 'Moderate distance to water'
                else:
                    if value > 0.6:
                        interpretation = 'High'
                    elif value < 0.3:
                        interpretation = 'Low'
                    else:
                        interpretation = 'Moderate'
                
                # Format value with appropriate precision
                if 'Soil Type' in feature or 'Wilderness' in feature:
                    value_str = '✓' if value > 0.5 else '✗'
                else:
                    value_str = f'{value:.3f}'
                
                table_data.append({
                    'Category': '',
                    'Feature Name': feature.replace('_', ' '),
                    'Value': value_str,
                    'Interpretation': interpretation
                })
    
    # Create conditional styling for group headers
    style_data_conditional = []
    for i, row in enumerate(table_data):
        if row['Category'] != '':
            style_data_conditional.append({
                'if': {'row_index': i},
                'backgroundColor': 'rgb(240, 240, 240)',
                'fontWeight': 'bold'
            })

    return dash_table.DataTable(
        data=table_data,
        columns=[
            {'name': 'Category', 'id': 'Category'},
            {'name': 'Feature Name', 'id': 'Feature Name'},
            {'name': 'Value', 'id': 'Value'},
            {'name': 'Interpretation', 'id': 'Interpretation'}
        ],
        style_cell={
            'textAlign': 'left',
            'padding': '8px',
            'fontFamily': 'Arial',
            'fontSize': '12px',
            'whiteSpace': 'normal',
            'height': 'auto'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'fontSize': '13px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': i for i in range(len(table_data)) if table_data[i]['Category'] != ''},
                'backgroundColor': 'rgb(240, 240, 240)',
                'fontWeight': 'bold'
            }
        ],
        style_table={
            'overflowX': 'auto', 
            'maxHeight': '500px',
            'border': '1px solid #ddd'
        },
        sort_action='none',
        page_size=60
    )

def create_soft_3d_plot():
    probs = gmm.predict_proba(latent_vectors)
    n_clusters = probs.shape[1]
    # Get cluster colours (tab10 works for up to 10 clusters)
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    cluster_colors = cmap(np.arange(n_clusters))[:, :3]   # (n_clusters, 3)
    point_colors = probs @ cluster_colors                # (n_samples, 3)
    point_colors_hex = [f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})' for c in point_colors]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=latent_vectors[:, 0],
        y=latent_vectors[:, 1],
        z=latent_vectors[:, 2],
        mode='markers',
        marker=dict(size=4, color=point_colors_hex, opacity=0.8),
        text=[f'Point {i}<br>Probabilities: {probs[i]}' for i in range(len(latent_vectors))],
        hoverinfo='text',
        name='Data points',
        customdata=latent_vectors
    ))
    # Add centroids
    fig.add_trace(go.Scatter3d(
        x=gmm.means_[:, 0],
        y=gmm.means_[:, 1],
        z=gmm.means_[:, 2],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond', line=dict(width=2, color='black')),
        name='Centroids'
    ))
    # Add a trace for the selected point (initially invisible)
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=15, color='yellow', symbol='circle', line=dict(width=2, color='black')),
        name='Selected point'
    ))
    fig.update_layout(
        title='Latent Space Explorer – Click on any point or use sliders',
        scene=dict(
            xaxis_title='Latent dim 1',
            yaxis_title='Latent dim 2',
            zaxis_title='Latent dim 3',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='cube'
        ),
        width=800, height=600
    )
    return fig

def heatmap_from_features(features):
    """Generate a stylised 1-row heatmap where each column is a feature"""
    fig, ax = plt.subplots(figsize=(16, 4))
    
    # Reshape to 1 row with all features
    reshaped = features.reshape(1, -1)
    n_features = len(features)
    
    # Create the heatmap
    im = ax.imshow(reshaped, cmap='RdYlBu_r', aspect='auto', interpolation='bilinear')
    
    # Set x-ticks for each feature
    ax.set_xticks(np.arange(n_features))
    
    # Create feature labels (shortened for readability)
    feature_labels = []
    for i, name in enumerate(feature_names):
        # Shorten long names for better display
        if 'Horizontal_Distance_To_Hydrology' in name:
            short_name = 'HydroDist'
        elif 'Vertical_Distance_To_Hydrology' in name:
            short_name = 'HydroVert'
        elif 'Horizontal_Distance_To_Roadways' in name:
            short_name = 'RoadDist'
        elif 'Wilderness_Area' in name:
            area = name.replace('Wilderness_Area_', '')
            short_name = f'Wild_{area[:4]}'
        elif 'Soil_Type' in name:
            soil_num = name.replace('Soil_Type_', '')
            short_name = f'Soil{soil_num}'
        else:
            short_name = name[:12]  # Truncate to 12 chars
        
        feature_labels.append(short_name)
    
    ax.set_xticklabels(feature_labels, rotation=90, fontsize=8)
    
    # No y-ticks needed for single row
    ax.set_yticks([])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Feature Value', shrink=0.8)
    cbar.ax.set_ylabel('Low ← Value → High', rotation=270, labelpad=20)
    
    ax.set_title(f'Generated Sample – Feature Heatmap\n(Red=High Value, Blue=Low Value, {n_features} Features)', 
                 fontsize=12, pad=20)
    
    # Add grid lines to separate features
    ax.set_xticks(np.arange(n_features) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f'data:image/png;base64,{encoded}'

def particle_from_features(features):
    """Generate a particle system Plotly figure (returns HTML div string)"""
    n = len(features)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    norm_vals = (features - features.min()) / (features.max() - features.min() + 1e-8)
    radii = 1 + norm_vals * 0.8
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    sizes = 10 + norm_vals * 30
    colors = norm_vals

    # Create hover text with full feature names
    hover_texts = []
    for i, name in enumerate(feature_names):
        if 'Soil Type' in name:
            val_str = 'Present' if features[i] > 0.5 else 'Absent'
        elif 'Wilderness' in name:
            val_str = 'Yes' if features[i] > 0.5 else 'No'
        else:
            val_str = f'{features[i]:.3f}'
        hover_texts.append(f"<b>{name}</b><br>Value: {val_str}<br>Normalised: {norm_vals[i]:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers+text',
        marker=dict(size=sizes, color=colors, colorscale='Viridis', showscale=True,
                    colorbar=dict(title='Feature value')),
        text=[f'F{i}' for i in range(n)],
        textposition='middle center',
        hoverinfo='text'
    ))
    # Add outer circle
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode='lines',
                             line=dict(color='gray', width=2, dash='dash'), showlegend=False))
    fig.update_layout(
        title='Particle System – Each particle represents a feature',
        xaxis=dict(visible=False, range=[-2.2, 2.2]),
        yaxis=dict(visible=False, range=[-2.2, 2.2]),
        width=500, height=500, showlegend=False
    )
    return dcc.Graph(figure=fig)

#DASH APP
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Generative Forest Cover Visualisation", className="text-center my-3"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='latent-plot', figure=create_soft_3d_plot(), style={'height': '550px'})
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Latent Space Controls"),
                dbc.CardBody([
                    html.Label("Dimension 1", className="fw-bold"),
                    dcc.Slider(id='z1-slider',
                               min=float(latent_vectors[:,0].min()), max=float(latent_vectors[:,0].max()),
                               step=0.05, value=float(latent_vectors[0,0]),
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Dimension 2", className="fw-bold"),
                    dcc.Slider(id='z2-slider',
                               min=float(latent_vectors[:,1].min()), max=float(latent_vectors[:,1].max()),
                               step=0.05, value=float(latent_vectors[0,1]),
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    html.Label("Dimension 3", className="fw-bold"),
                    dcc.Slider(id='z3-slider',
                               min=float(latent_vectors[:,2].min()), max=float(latent_vectors[:,2].max()),
                               step=0.05, value=float(latent_vectors[0,2]),
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.Br(),
                    dbc.Button("Random Point", id="random-btn", color="primary", className="mt-2 w-100")
                ])
            ], className="mb-3")
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Visualisation Mode"),
                dbc.CardBody([
                    dcc.RadioItems(id='viz-mode',
                                   options=[
                                       {'label': 'Heatmap View', 'value': 'heatmap'},
                                       {'label': 'Particle System', 'value': 'particle'},
                                       {'label': 'Feature Table', 'value': 'table'}
                                   ],
                                   value='heatmap', inline=True, className="mb-3"),
                    html.Div(id='generated-viz')
                ])
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Interpretation Guide"),
                dbc.CardBody([
                    html.P("Colour Meaning:", className="fw-bold"),
                    html.Ul([
                        html.Li("Red/Warm = High feature value (more prevalent)", className="small"),
                        html.Li("Blue/Cool = Low feature value (less prevalent)", className="small"),
                    ]),
                    html.Hr(),
                    html.P("Feature Table:", className="fw-bold"),
                    html.P("Switch to 'Feature Table' mode to see detailed information about all 52 features", className="small"),
                    html.Hr(),
                    html.P("Tip: Hover over particles in the particle system to see feature names and values!", className="small text-muted")
                ])
            ])
        ], width=4)
    ])
], fluid=True)

@app.callback(
    [Output('z1-slider', 'value'),
     Output('z2-slider', 'value'),
     Output('z3-slider', 'value')],
    [Input('latent-plot', 'clickData'),
     Input('random-btn', 'n_clicks')],
    [State('z1-slider', 'value'),
     State('z2-slider', 'value'),
     State('z3-slider', 'value')]
)
def update_sliders_from_click(clickData, n_clicks, z1, z2, z3):
    ctx = callback_context
    if not ctx.triggered:
        return [z1, z2, z3]
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == 'latent-plot' and clickData:
        # Handle different Plotly clickData formats
        try:
            # Try to get point index from different possible keys
            if 'points' in clickData and len(clickData['points']) > 0:
                point = clickData['points'][0]
                
                # Check for different possible index keys
                if 'pointIndex' in point:
                    point_idx = point['pointIndex']
                elif 'pointNumber' in point:
                    point_idx = point['pointNumber']
                elif 'customdata' in point:
                    # If customdata contains the index, use that
                    point_idx = point['customdata']
                else:
                    print(f"Could not find point index in clickData: {point.keys()}")
                    return [z1, z2, z3]
                
                # Get the latent vector for this point
                if point_idx < len(latent_vectors):
                    z = latent_vectors[point_idx]
                    return [z[0], z[1], z[2]]
                else:
                    print(f"Point index {point_idx} out of range for latent_vectors")
                    return [z1, z2, z3]
            else:
                print("No points found in clickData")
                return [z1, z2, z3]
                
        except Exception as e:
            print(f"Error processing clickData: {e}")
            print(f"clickData structure: {clickData}")
            return [z1, z2, z3]
    
    elif trigger_id == 'random-btn':
        idx = np.random.randint(0, len(latent_vectors))
        z = latent_vectors[idx]
        return [z[0], z[1], z[2]]
    
    return [z1, z2, z3]

@app.callback(
    Output('generated-viz', 'children'),
    [Input('z1-slider', 'value'),
     Input('z2-slider', 'value'),
     Input('z3-slider', 'value'),
     Input('viz-mode', 'value')]
)
def update_generated(z1, z2, z3, mode):
    z = np.array([z1, z2, z3])
    features = decode_latent(z)
    if mode == 'heatmap':
        img_data = heatmap_from_features(features)
        return html.Img(src=img_data, style={'width': '100%'})
    elif mode == 'particle':
        return html.Div([
            particle_from_features(features),
            html.Div([
                html.P("Each particle represents one of the 52 features.", className="small text-muted mt-2"),
                html.P("Distance from centre = Feature prominence. Hover over particles to see feature names!", className="small text-muted")
            ])
        ])
    else:  # table mode
        return html.Div([
            html.H6("Complete Feature Breakdown", className="mb-2"),
            create_feature_table(features),
            html.Div([
                html.P("This table shows the actual values and interpretations for all 52 features.", className="small text-muted mt-2"),
                html.P("• ✅ = Feature present, ❌ = Feature absent", className="small text-muted"),
                html.P("• 🔥 = High value, 💙 = Low value, ⚖️ = Moderate value", className="small text-muted")
            ])
        ])


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)