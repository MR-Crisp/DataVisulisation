import numpy as np
from shapely.geometry import Polygon, box
from scipy.spatial import Voronoi
import plotly.graph_objects as go


def voronoi_finite_polygons(vor, bounding_box_margin=0.5):
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
            poly_verts = vor.vertices[region]
        else:
            ridges = [
                (p1, p2, v1, v2)
                for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices)
                if point_idx in (p1, p2)
            ]
            verts = []
            for p1, p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    verts.append(vor.vertices[v1])
                    verts.append(vor.vertices[v2])
                else:
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

    class_stats = {}
    for cls in unique_classes:
        mask = labels == cls
        pts = coords_2d[mask]
        class_stats[cls] = {
            "count": int(mask.sum()),
            "pct": float(mask.sum() / len(labels) * 100),
            "centroid": pts.mean(axis=0) if len(pts) > 0 else np.array([0, 0])
        }


    def parse_colour(colour):
        if colour.startswith('rgb'):
            nums = colour.replace('rgb(', '').replace(')', '').split(',')
            return int(nums[0]), int(nums[1]), int(nums[2])
        else:
            return int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)

    for cls in unique_classes:
        colour = class_colour[cls]
        label = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
        r, g, b = parse_colour(colour)
        fill_rgba = f"rgba({r},{g},{b},0.25)"
        line_rgba = f"rgba({r},{g},{b},0.7)"

        x_all, y_all = [], []
        for i, poly in enumerate(polygons):
            if poly is None or labels[i] != cls:
                continue
            x_all.extend(poly[:, 0].tolist() + [None])
            y_all.extend(poly[:, 1].tolist() + [None])

        if not x_all:
            continue

        stats = class_stats[cls]
        fig.add_trace(go.Scatter(
            x=x_all,
            y=y_all,
            fill='toself',
            fillcolor=fill_rgba,
            line=dict(color=line_rgba, width=0.8),
            mode='lines',
            name=label,
            legendgroup=str(cls),
            showlegend=True,
            hoverinfo='skip',
        ))

    for cls in unique_classes:
        mask = labels == cls
        label = class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
        stats = class_stats[cls]
        colour = class_colour[cls]
        r, g, b = parse_colour(colour)

        fig.add_trace(go.Scatter(
            x=coords_2d[mask, 0].tolist(),
            y=coords_2d[mask, 1].tolist(),
            mode='markers',
            marker=dict(
                color=f"rgb({r},{g},{b})",
                size=4,
                opacity=0.8,
                line=dict(width=0.3, color='white')
            ),
            name=label,
            legendgroup=str(cls),
            showlegend=False,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Count: {stats['count']} ({stats['pct']:.1f}%)<br>"
                "X: %{x:.3f}<br>"
                "Y: %{y:.3f}<br>"
                "<extra></extra>"
            )
        ))

    cx = [class_stats[cls]["centroid"][0] for cls in unique_classes]
    cy = [class_stats[cls]["centroid"][1] for cls in unique_classes]
    centroid_labels = [
        class_names.get(cls, f"Class {cls}") if class_names else f"Class {cls}"
        for cls in unique_classes
    ]
    centroid_hover = [
        f"<b>{centroid_labels[i]} — centroid</b><br>"
        f"Count: {class_stats[cls]['count']} ({class_stats[cls]['pct']:.1f}%)<br>"
        f"X: {cx[i]:.3f}<br>Y: {cy[i]:.3f}<extra></extra>"
        for i, cls in enumerate(unique_classes)
    ]

    fig.add_trace(go.Scatter(
        x=cx, y=cy,
        mode='markers+text',
        marker=dict(
            symbol='star',
            size=14,
            color=[class_colour[cls] for cls in unique_classes],
            line=dict(width=1.5, color='black'),
            opacity=1.0
        ),
        text=centroid_labels,
        textposition='top center',
        textfont=dict(size=9, color='black'),
        hovertemplate=centroid_hover,
        name='Class centroids',
        showlegend=True,
    ))

    n_classes = len(unique_classes)
    total_points = len(labels)

    fig.update_layout(
        title=dict(
            text=f"Voronoi — UMAP Latent Space  |  {n_classes} classes  |  {total_points} points",
            font=dict(size=14)
        ),
        plot_bgcolor='#F8F8F8',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(
            title="Classes",
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='lightgray',
            borderwidth=1,
        ),
        xaxis=dict(
            title="UMAP Dimension 1",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.4)',
            zeroline=False,
        ),
        yaxis=dict(
            title="UMAP Dimension 2",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.4)',
            zeroline=False,
        ),
        hovermode='closest',
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig