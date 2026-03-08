"""
data_loader.py
--------------
Fetchers for INEGI and SEP open government datasets.
Handles downloading, caching, and basic validation.
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

INEGI_API_KEY = os.getenv("INEGI_API_KEY", "")

# ---------------------------------------------------------------------------
# INEGI Census 2020
# ---------------------------------------------------------------------------

INEGI_CENSUS_URL = (
    "https://www.inegi.org.mx/contenidos/programas/ccpv/2020/datosabiertos/"
    "iter/iter_00_cpv2020_csv.zip"
)

CENSUS_COLUMNS = {
    "ENTIDAD": "state_code",
    "MUN": "municipality_code",
    "NOM_MUN": "municipality_name",
    "POBTOT": "total_population",
    "P_18YMAS": "pop_18_and_over",
    "GRAPROES": "avg_schooling_years",
    "PROM_HNV": "avg_children_born",
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> Path:
    """Download a file with progress bar, skip if already cached."""
    if dest_path.exists():
        logger.info(f"Cache hit: {dest_path.name}")
        return dest_path

    logger.info(f"Downloading {url} → {dest_path.name}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in response.iter_content(chunk_size):
            f.write(chunk)
            bar.update(len(chunk))

    return dest_path


def load_inegi_census(state_filter: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Load INEGI 2020 Census municipal-level data.

    Parameters
    ----------
    state_filter : list of state codes (e.g. ['09'] for CDMX), or None for all.

    Returns
    -------
    pd.DataFrame with renamed columns and numeric types enforced.
    """
    zip_path = RAW_DATA_DIR / "iter_00_cpv2020.zip"
    download_file(INEGI_CENSUS_URL, zip_path)

    df = pd.read_csv(
        zip_path,
        compression="zip",
        encoding="latin-1",
        low_memory=False,
        usecols=list(CENSUS_COLUMNS.keys()),
        dtype={"ENTIDAD": str, "MUN": str},
    )
    df = df.rename(columns=CENSUS_COLUMNS)

    # Filter to municipal level (exclude state totals where MUN == '000')
    df = df[df["municipality_code"] != "000"].copy()

    # Enforce numeric
    for col in ["total_population", "pop_18_and_over", "avg_schooling_years"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if state_filter:
        df = df[df["state_code"].isin(state_filter)]

    logger.info(f"Census loaded: {len(df):,} municipalities")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# SEP / ANUIES Higher Education Stats
# ---------------------------------------------------------------------------

ANUIES_URL = (
    "https://www.anuies.mx/informacion-y-servicios/informacion-estadistica-de-educacion-superior/"
    "anuario-estadistico-de-educacion-superior"
)

# Fallback: direct CSV links that ANUIES publishes annually
ANUIES_POSGRADO_URLS = {
    2022: "https://www.anuies.mx/content/anuarios/2022/posgrado/posgrado_2022.csv",
    2021: "https://www.anuies.mx/content/anuarios/2021/posgrado/posgrado_2021.csv",
}


def load_sep_posgrado(year: int = 2022) -> pd.DataFrame:
    """
    Load SEP/ANUIES postgraduate enrollment statistics.

    Returns DataFrame with columns:
        institution, state, program, level, enrollment_total, graduates_total
    """
    url = ANUIES_POSGRADO_URLS.get(year)
    if not url:
        raise ValueError(f"No URL configured for year {year}. Add it to ANUIES_POSGRADO_URLS.")

    cache_path = RAW_DATA_DIR / f"anuies_posgrado_{year}.csv"

    try:
        download_file(url, cache_path)
        df = pd.read_csv(cache_path, encoding="latin-1")
    except requests.HTTPError:
        logger.warning(
            "Could not download ANUIES CSV directly. "
            "Visit https://www.anuies.mx and download manually to data/raw/."
        )
        # Return synthetic placeholder for development
        df = _synthetic_posgrado_placeholder(year)

    return df


def _synthetic_posgrado_placeholder(year: int) -> pd.DataFrame:
    """
    Synthetic data matching ANUIES schema for offline development.
    Replace with real data once manually downloaded.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    states = [f"{i:02d}" for i in range(1, 33)]
    records = []
    for state in states:
        n_institutions = rng.integers(5, 30)
        for _ in range(n_institutions):
            records.append({
                "year": year,
                "state_code": state,
                "institution": f"Institution_{rng.integers(1000)}",
                "level": rng.choice(["Maestría", "Doctorado", "Especialidad"]),
                "enrollment_total": rng.integers(10, 500),
                "graduates_total": rng.integers(1, 100),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# INEGI DENUE (institution locations)
# ---------------------------------------------------------------------------

DENUE_BASE_URL = "https://www.inegi.org.mx/app/api/denue/v1/consulta"


def query_denue_universities(lat: float, lon: float, radius_meters: int = 5000) -> pd.DataFrame:
    """
    Query INEGI DENUE API for universities near a point.

    Requires INEGI_API_KEY in .env

    Parameters
    ----------
    lat, lon : coordinates
    radius_meters : search radius

    Returns
    -------
    GeoDataFrame of institutions with geometry column
    """
    if not INEGI_API_KEY:
        raise EnvironmentError("INEGI_API_KEY not set. Get one at https://www.inegi.org.mx/app/api/denue/")

    # SCIAN code 611310 = Universidades e instituciones de educación superior
    url = (
        f"{DENUE_BASE_URL}/buscarrea/"
        f"{lat},{lon}/{radius_meters}/611310/json/{INEGI_API_KEY}"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not data:
        return gpd.GeoDataFrame()

    df = pd.DataFrame(data)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["Longitud"].astype(float),
            df["Latitud"].astype(float)
        ),
        crs="EPSG:4326"
    )
    return gdf


# ---------------------------------------------------------------------------
# Municipality Boundaries (GeoJSON)
# ---------------------------------------------------------------------------

MX_MUNICIPALITIES_URL = (
    "https://raw.githubusercontent.com/angelnmara/geojson/master/"
    "mexicoHigh.json"  # State level; for municipalities use INEGI's MGN
)

INEGI_MGN_URL = (
    "https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/"
    "bvinegi/productos/geografia/marcogeo/889463770541_s.zip"
)


def load_mexico_municipalities() -> gpd.GeoDataFrame:
    """Load Mexico municipality boundaries from INEGI Marco Geoestadístico Nacional."""
    cache_path = RAW_DATA_DIR / "mexico_municipalities.gpkg"

    if cache_path.exists():
        return gpd.read_file(cache_path)

    logger.info("Downloading Mexico MGN municipality boundaries...")
    zip_path = RAW_DATA_DIR / "mgn_municipalities.zip"

    try:
        download_file(INEGI_MGN_URL, zip_path)
        gdf = gpd.read_file(f"zip://{zip_path}!mun.shp")
    except Exception as e:
        logger.warning(f"Could not load INEGI MGN: {e}. Using simplified fallback.")
        gdf = _load_simplified_municipalities()

    gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(cache_path, driver="GPKG")
    return gdf


def _load_simplified_municipalities() -> gpd.GeoDataFrame:
    """Fallback: load simplified Mexico state boundaries."""
    url = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
    return gpd.read_file(url)
