import numpy as np
import base64
import io
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib
import matplotlib.pyplot as plt

FEATURE_DEFINITIONS = {
    'Elevation': 0, 'Aspect': 1, 'Slope': 2,
    'Horizontal_Distance_To_Hydrology': 3, 'Vertical_Distance_To_Hydrology': 4,
    'Horizontal_Distance_To_Roadways': 5, 'Hillshade_9am': 6, 'Hillshade_Noon': 7, 'Hillshade_3pm': 8,
    'Wilderness_Area_Rawah': 9, 'Wilderness_Area_Neota': 10, 'Wilderness_Area_Comanche': 11,
    'Wilderness_Area_Cache_La_Poudre': 12,
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
feature_names = [name for name, idx in sorted(FEATURE_DEFINITIONS.items(), key=lambda x: x[1])]

# ---- Heatmap ----
def compute_heatmap_data(features):
    n_features = len(features)
    short_labels = []
    
    # Shorten feature names for display
    short_labels = []
    for name in feature_names[:n_features]:
        if 'Horizontal_Distance_To_Hydrology' in name:
            short = 'HydroDist'
        elif 'Vertical_Distance_To_Hydrology' in name:
            short = 'HydroVert'
        elif 'Horizontal_Distance_To_Roadways' in name:
            short = 'RoadDist'
        elif 'Wilderness_Area' in name:
            area = name.replace('Wilderness_Area_', '')
            short = f'Wild_{area[:4]}'
        elif 'Soil_Type' in name:
            num = name.replace('Soil_Type_', '')
            short = f'Soil{num}'
        else:
            short = name[:12]
        short_labels.append(short)
    
    # Normalise values to [0,1] for better colour mapping
    min_val = features.min()
    max_val = features.max()
    norm_values = (features - min_val) / (max_val - min_val + 1e-8)
    
    return {
        'feature_names': short_labels,
        'values': features.tolist(),
        'norm_values': norm_values.tolist(),
        'min_value': float(min_val),
        'max_value': float(max_val)
    }