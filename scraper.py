"""
scraper.py
----------
Web scrapers for collecting supplementary education data from public sources.
Targets SEP, CONACYT, and ANUIES open portals.
"""

import logging
import time
import re
from pathlib import Path
from typing import Optional

import requests
import httpx
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ProdigyResearchBot/1.0; "
        "educational research project)"
    )
}


# ---------------------------------------------------------------------------
# CONACYT Postgraduate Programs Registry
# ---------------------------------------------------------------------------

CONACYT_PNPC_URL = "https://www.conahcyt.mx/siicyt/index.php/estadisticas-nacionales"


def scrape_conacyt_pnpc_stats() -> pd.DataFrame:
    """
    Scrape CONACYT/CONAHCYT PNPC (Programa Nacional de Posgrado de Calidad)
    program statistics.

    Returns DataFrame with program name, institution, level, state, enrollment.
    Note: CONAHCYT restructured the registry in 2023 — fall back to CSV download
    if dynamic content fails.
    """
    logger.info(f"Fetching CONAHCYT PNPC stats from {CONACYT_PNPC_URL}")

    try:
        response = requests.get(CONACYT_PNPC_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        # Look for downloadable CSV/Excel links
        csv_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if any(ext in href.lower() for ext in [".csv", ".xlsx", ".xls"]):
                csv_links.append(href)

        if csv_links:
            logger.info(f"Found {len(csv_links)} data file links")
            return _download_and_parse_csv(csv_links[0])

    except requests.RequestException as e:
        logger.warning(f"Could not scrape CONAHCYT: {e}")

    return _fallback_pnpc_data()


def _download_and_parse_csv(url: str) -> pd.DataFrame:
    """Download and parse a CSV from a URL."""
    if not url.startswith("http"):
        url = "https://www.conahcyt.mx" + url

    cache_path = RAW_DATA_DIR / f"pnpc_{hash(url) % 99999}.csv"
    if not cache_path.exists():
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)

    try:
        return pd.read_csv(cache_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(cache_path, encoding="latin-1")


def _fallback_pnpc_data() -> pd.DataFrame:
    """Return structured placeholder when live scraping fails."""
    logger.warning("Using fallback PNPC data — replace with real data from CONAHCYT")
    return pd.DataFrame({
        "program": ["Ejemplo Maestría", "Ejemplo Doctorado"],
        "institution": ["UNAM", "IPN"],
        "level": ["Maestría", "Doctorado"],
        "state": ["CDMX", "CDMX"],
        "enrolled": [150, 80],
        "graduates": [45, 20],
        "note": ["synthetic", "synthetic"],
    })


# ---------------------------------------------------------------------------
# SEP Statistics Bulletin Scraper
# ---------------------------------------------------------------------------

SEP_BULLETIN_BASE = "https://www.planeacion.sep.gob.mx/principalescifras/"


def scrape_sep_principal_cifras(year: int = 2023) -> Optional[pd.DataFrame]:
    """
    Scrape SEP 'Principales Cifras' education statistics bulletin.
    These are published annually as PDF/Excel downloads.

    Returns DataFrame or None if unavailable.
    """
    logger.info(f"Scraping SEP Principales Cifras for {year}")

    try:
        response = requests.get(SEP_BULLETIN_BASE, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        # Find links containing the target year
        year_links = [
            a["href"] for a in soup.find_all("a", href=True)
            if str(year) in a.get_text() or str(year) in a["href"]
        ]

        if not year_links:
            logger.warning(f"No SEP bulletin links found for {year}")
            return None

        # Prefer Excel downloads
        excel_links = [l for l in year_links if ".xlsx" in l.lower() or ".xls" in l.lower()]
        target = excel_links[0] if excel_links else year_links[0]

        if not target.startswith("http"):
            target = "https://www.planeacion.sep.gob.mx" + target

        cache_path = RAW_DATA_DIR / f"sep_cifras_{year}.xlsx"
        if not cache_path.exists():
            resp = requests.get(target, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)

        return pd.read_excel(cache_path, sheet_name=None)  # all sheets

    except Exception as e:
        logger.warning(f"SEP scraper failed: {e}")
        return None


# ---------------------------------------------------------------------------
# INEGI Data Portal Scraper (dataset discovery)
# ---------------------------------------------------------------------------

INEGI_DATOS_URL = "https://datos.gob.mx/busca/dataset"


def discover_inegi_education_datasets(query: str = "educación superior posgrado") -> list[dict]:
    """
    Search datos.gob.mx for education-related datasets.

    Returns list of {title, url, description, format} dicts.
    """
    params = {"q": query, "tags": "educacion"}
    try:
        resp = requests.get(INEGI_DATOS_URL, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        datasets = []
        for result in soup.select(".dataset-item, .search-result"):
            title_el = result.select_one("h3, .dataset-heading")
            desc_el = result.select_one(".notes, .dataset-description")
            link_el = result.select_one("a[href]")

            if title_el and link_el:
                datasets.append({
                    "title": title_el.get_text(strip=True),
                    "description": desc_el.get_text(strip=True) if desc_el else "",
                    "url": link_el["href"],
                })

        logger.info(f"Found {len(datasets)} datasets for query: '{query}'")
        return datasets

    except Exception as e:
        logger.warning(f"Dataset discovery failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Rate-limited async fetcher for bulk requests
# ---------------------------------------------------------------------------

async def fetch_many(
    urls: list[str],
    delay_seconds: float = 1.0,
    timeout: float = 30.0,
) -> list[Optional[str]]:
    """
    Async rate-limited fetcher for multiple URLs.
    Respects robots.txt courtesy delay.
    """
    results = []
    async with httpx.AsyncClient(headers=HEADERS, timeout=timeout) as client:
        for i, url in enumerate(urls):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                results.append(resp.text)
                logger.debug(f"[{i+1}/{len(urls)}] OK: {url}")
            except httpx.HTTPError as e:
                logger.warning(f"[{i+1}/{len(urls)}] FAILED: {url} — {e}")
                results.append(None)

            if i < len(urls) - 1:
                await __import__("asyncio").sleep(delay_seconds)

    return results


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Discovering education datasets ===")
    datasets = discover_inegi_education_datasets()
    for ds in datasets[:3]:
        print(f"  - {ds['title']}: {ds['url']}")

    print("\n=== Scraping CONAHCYT PNPC ===")
    df = scrape_conacyt_pnpc_stats()
    print(df.head())
