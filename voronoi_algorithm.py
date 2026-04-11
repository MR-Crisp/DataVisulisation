import numpy as np
from shapely.geometry import Polygon, box
from scipy.spatial import Voronoi
import plotly.graph_objects as go


# ── 3. Clip infinite Voronoi regions to a bounding box ────────────────────
def voronoi_finite_polygons(vor, bounding_box_margin=0.5):
    """
    Reconstruct infinite Voronoi regions as finite polygons by clipping
    against a bounding box slightly larger than the point cloud.
    Returns: list of (region_vertices_array | None) per seed point.
    """
    pts = vor.points
    pad = bounding_box_margin
    clip_box = box(
        pts[:, 0].min() - pad, pts[:, 1].min() - pad,
        pts[:, 0].max() + pad, pts[:, 1].max() + pad
    )

    center = pts.mean(axis=0)
    polygons = []

    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]

        if not region:
            polygons.append(None)
            continue

        if -1 not in region:
            # Fully finite region — use vertices directly
            poly_verts = vor.vertices[region]
        else:
            # Infinite region — reconstruct missing vertices by pushing them
            # outward from the centre along the ridge direction
            ridges = [
                (p1, p2, v1, v2)
                for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices)
                if point_idx in (p1, p2)
            ]
            verts = []
            for p1, p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1  # ensure v1 is the infinite end
                if v1 >= 0:
                    verts.append(vor.vertices[v1])
                    verts.append(vor.vertices[v2])
                else:
                    # Compute the far point
                    tangent = pts[p2] - pts[p1]
                    tangent /= np.linalg.norm(tangent)
                    normal = np.array([-tangent[1], tangent[0]])
                    midpoint = pts[[p1, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, normal)) * normal
                    far_point = vor.vertices[v2] + direction * 1e3
                    verts.append(vor.vertices[v2])
                    verts.append(far_point)

            if len(verts) < 3:
                polygons.append(None)
                continue
            poly_verts = np.array(verts)

        # Clip to bounding box using shapely
        try:
            poly = Polygon(poly_verts).convex_hull.intersection(clip_box)
            if poly.is_empty or not poly.is_valid:
                polygons.append(None)
            else:
                polygons.append(np.array(poly.exterior.coords))
        except Exception:
            polygons.append(None)

    return polygons

def plot_voronoi(coords_2d, labels, class_colour, class_names=None):
    vor = Voronoi(coords_2d)
    polygons = voronoi_finite_polygons(vor, bounding_box_margin=1.0)

    fig = go.Figure()

    unique_classes = np.unique(labels)
    for cls in unique_classes:
        colour = class_colour[cls]
        label = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"

        if colour.startswith('rgb'):
            nums = colour.replace('rgb(', '').replace(')', '').split(',')
            r, g, b = int(nums[0]), int(nums[1]), int(nums[2])
        else:
            r, g, b = int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)

        fill_rgba = f"rgba({r},{g},{b},0.35)"
        line_rgba = f"rgba({r},{g},{b},0.6)"

        x_all, y_all = [], []
        for i, poly in enumerate(polygons):
            if poly is None:
                continue
            if labels[i] != cls:
                continue
            x_all.extend(poly[:, 0].tolist() + [None])
            y_all.extend(poly[:, 1].tolist() + [None])

        if not x_all:
            continue

        fig.add_trace(go.Scatter(
            x=x_all,
            y=y_all,
            fill='toself',
            fillcolor=fill_rgba,
            line=dict(color=line_rgba, width=0.5),
            mode='lines',
            name=label,
            legendgroup=str(cls),
            showlegend=True,
            hoverinfo='skip'
        ))

    for cls in unique_classes:
        mask = labels == cls
        label = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
        fig.add_trace(go.Scatter(
            x=coords_2d[mask, 0].tolist(),
            y=coords_2d[mask, 1].tolist(),
            mode='markers',
            marker=dict(color=class_colour[cls], size=3, opacity=0.7),
            name=label,
            legendgroup=str(cls),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig.update_layout(
        title="Voronoi diagram — UMAP latent space",
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
    )

    return fig