"""
geo_utils.py
------------
Geospatial helper functions for radius queries, choropleth prep, and map generation.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
from shapely.geometry import Point, mapping
from shapely.ops import transform
import pyproj

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "maps"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Danish Embassy, Mexico City
DANISH_EMBASSY_CDMX = {"lat": 19.4284, "lon": -99.1927, "name": "Danish Embassy, CDMX"}


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two lat/lon points in kilometers."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def geodesic_buffer(lat: float, lon: float, radius_km: float) -> gpd.GeoDataFrame:
    """
    Create a geodesically accurate circular buffer around a point.

    Returns a GeoDataFrame in EPSG:4326.
    """
    # Project to UTM for accurate buffering
    point_wgs84 = Point(lon, lat)
    utm_zone = int((lon + 180) / 6) + 1
    utm_crs = pyproj.CRS(f"+proj=utm +zone={utm_zone} +datum=WGS84")
    wgs84_crs = pyproj.CRS("EPSG:4326")

    project_to_utm = pyproj.Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True).transform
    project_to_wgs84 = pyproj.Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True).transform

    point_utm = transform(project_to_utm, point_wgs84)
    buffer_utm = point_utm.buffer(radius_km * 1000)
    buffer_wgs84 = transform(project_to_wgs84, buffer_utm)

    return gpd.GeoDataFrame(
        {"geometry": [buffer_wgs84], "radius_km": [radius_km]},
        crs="EPSG:4326"
    )


def points_in_radius(
    gdf: gpd.GeoDataFrame,
    lat: float,
    lon: float,
    radius_km: float,
) -> gpd.GeoDataFrame:
    """Filter a GeoDataFrame to points within radius_km of (lat, lon)."""
    buffer = geodesic_buffer(lat, lon, radius_km)
    return gdf[gdf.geometry.within(buffer.geometry.iloc[0])].copy()


def compute_municipality_area_km2(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add area_km2 column to a municipality GeoDataFrame."""
    projected = gdf.to_crs("EPSG:6372")  # Mexico ITRF2008
    gdf = gdf.copy()
    gdf["area_km2"] = projected.geometry.area / 1e6
    return gdf


# ---------------------------------------------------------------------------
# Folium Map Builders
# ---------------------------------------------------------------------------

def build_choropleth_map(
    municipalities_gdf: gpd.GeoDataFrame,
    value_column: str = "prob_at_least_one",
    center_lat: float = 23.6345,
    center_lon: float = -102.5528,
    zoom: int = 5,
    title: str = "P(Prodigy) by Municipality",
) -> folium.Map:
    """
    Build a choropleth folium map of prodigy probability by municipality.

    Parameters
    ----------
    municipalities_gdf : GeoDataFrame with geometry + value_column
    value_column : column to choropleth on
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )

    folium.Choropleth(
        geo_data=municipalities_gdf.__geo_interface__,
        data=municipalities_gdf,
        columns=["municipality_name", value_column],
        key_on="feature.properties.municipality_name",
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.2,
        legend_name=title,
        nan_fill_color="lightgrey",
    ).add_to(m)

    # Tooltip
    folium.GeoJson(
        municipalities_gdf,
        tooltip=folium.GeoJsonTooltip(
            fields=["municipality_name", value_column, "population"],
            aliases=["Municipality", "P(Prodigy)", "Population"],
            localize=True,
        ),
        style_function=lambda x: {"fillOpacity": 0, "weight": 0},
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def build_radius_map(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    prob_at_least_one: float,
    expected_count: float,
    label: str = "Query Point",
    municipalities_gdf: Optional[gpd.GeoDataFrame] = None,
) -> folium.Map:
    """
    Build a folium map showing a radius query around a point.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron",
    )

    # Draw radius circle
    folium.Circle(
        location=[center_lat, center_lon],
        radius=radius_km * 1000,
        color="#e63946",
        fill=True,
        fill_opacity=0.15,
        popup=folium.Popup(
            f"<b>{label}</b><br>"
            f"Radius: {radius_km:.1f} km<br>"
            f"P(≥1 prodigy): {prob_at_least_one:.2e}<br>"
            f"Expected count: {expected_count:.4f}",
            max_width=250,
        ),
    ).add_to(m)

    # Center marker
    folium.Marker(
        location=[center_lat, center_lon],
        popup=label,
        icon=folium.Icon(color="red", icon="star"),
    ).add_to(m)

    # Optionally overlay municipality boundaries
    if municipalities_gdf is not None:
        folium.GeoJson(
            municipalities_gdf,
            style_function=lambda x: {
                "fillOpacity": 0,
                "color": "#457b9d",
                "weight": 1.5,
            },
        ).add_to(m)

    return m


def save_map(m: folium.Map, filename: str) -> Path:
    """Save a folium map to the outputs/maps directory."""
    path = OUTPUTS_DIR / filename
    m.save(str(path))
    logger.info(f"Map saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

def probability_heatmap_data(
    municipalities_gdf: gpd.GeoDataFrame,
    value_column: str = "prob_at_least_one",
) -> list[list[float]]:
    """
    Extract [lat, lon, weight] triples from municipality centroids for HeatMap.
    """
    centroids = municipalities_gdf.copy()
    centroids["centroid"] = municipalities_gdf.geometry.centroid
    return [
        [row["centroid"].y, row["centroid"].x, row[value_column]]
        for _, row in centroids.iterrows()
        if pd.notna(row[value_column])
    ]


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo: buffer around Danish Embassy
    emb = DANISH_EMBASSY_CDMX
    print(f"Geodesic buffer around {emb['name']}:")
    buf = geodesic_buffer(emb["lat"], emb["lon"], radius_km=1.0)
    print(f"  Area: {buf.geometry.area.iloc[0] * 1e10:.2f} km² (approx)")

    # Demo: distance check
    d = haversine_distance_km(emb["lat"], emb["lon"], 19.4326, -99.1332)
    print(f"  Distance to Zócalo: {d:.2f} km")
