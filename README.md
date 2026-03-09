# 🎓 Prodigy Geographic Analysis
### Estimating the Probability of Extreme Academic Acceleration Across Geographic Regions

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: INEGI](https://img.shields.io/badge/Data-INEGI%20Open-green)](https://www.inegi.org.mx/)
[![Data: SEP](https://img.shields.io/badge/Data-SEP%20Open-green)](https://www.sep.gob.mx/)

---

## 📌 Project Overview

This project uses **open government datasets**, **geospatial analysis**, and **Bayesian probability modeling** to estimate the likelihood of finding individuals who complete graduate-level education at age 17 or younger — broken down by Mexican municipality, city, and arbitrary geographic radius.

The motivating question:
> *"How many 17-year-olds finish graduate studies in Mexico, and what is the probability of one living within 1km of a given point?"*

This is a rigorous data science exercise combining:
- 🗺️ **Geospatial mapping** with `folium` and `geopandas`
- 📊 **Statistical & Bayesian modeling** with `scipy` and `pymc`
- 🌐 **Web scraping** of SEP and INEGI open portals
- 📈 **Interactive visualization** with `plotly`

---

## 🗂️ Project Structure

```
prodigy-geographic-analysis/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                  # Downloaded government datasets
│   └── processed/            # Cleaned, merged datasets
├── notebooks/
│   ├── 01_data_collection.ipynb        # Scraping & downloading open data
│   ├── 02_exploratory_analysis.ipynb   # EDA on education & census data
│   ├── 03_probability_modeling.ipynb   # Bayesian probability estimation
│   └── 04_geographic_visualization.ipynb  # Maps & spatial analysis
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # INEGI/SEP dataset fetchers
│   ├── scraper.py            # Web scraping utilities
│   ├── model.py              # Probability models
│   └── geo_utils.py          # Geospatial helper functions
├── outputs/
│   └── maps/                 # Exported HTML/PNG maps
├── tests/
│   ├── test_model.py
│   └── test_geo_utils.py
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 📦 Data Sources

| Source | Dataset | Format | URL |
|--------|---------|--------|-----|
| INEGI | Population & Housing Census 2020 | CSV/SHP | [inegi.org.mx](https://www.inegi.org.mx/programas/ccpv/2020/) |
| SEP | Higher Education Statistics (ANUIES) | CSV | [planeacion.sep.gob.mx](https://planeacion.sep.gob.mx) |
| INEGI | DENUE Business/Institution Directory | API | [inegi.org.mx/app/api/denue](https://www.inegi.org.mx/app/api/denue/) |
| CONACYT | Postgraduate Registry | CSV | [conacyt.mx](https://conacyt.mx) |
| OpenStreetMap | Mexico geoboundaries | GeoJSON | Via `osmnx` |

---

## 🔬 Methodology

### 1. Base Rate Estimation
Using SEP's ANUIES postgraduate enrollment data filtered by age cohorts, we estimate the national base rate of graduate completions under age 18.

### 2. Spatial Distribution Modeling
Population density from INEGI Census 2020 is used to distribute the base rate across municipalities using a **Poisson spatial model**.

### 3. Radius-Based Probability
Given a lat/lon and radius (default: 1km), we compute:

```
P(at least 1 prodigy in radius) = 1 - e^(-λ * A_ratio)
```

Where:
- `λ` = expected count in the municipality
- `A_ratio` = area of circle / area of municipality

### 4. Bayesian Uncertainty Quantification
We use `PyMC` to place priors on the base rate and propagate uncertainty through the spatial model, producing **credible intervals** rather than point estimates.

---

## 🚀 Quickstart

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/prodigy-geographic-analysis.git
cd prodigy-geographic-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your INEGI API key to .env

# Run notebooks in order
jupyter lab
```

---

## 📊 Sample Output

The pipeline produces:
- **Choropleth map** of estimated prodigy probability by municipality
- **Radius query tool**: drop a pin, get a probability estimate + confidence interval
- **Time series** of graduate enrollment by age group (2010–2022)
- **Comparative analysis**: Mexico vs. OECD countries

---
Transparency International Corruption Perception Index table: https://www.transparency.org/en/cpi/2025

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📄 License


---

## 🙏 Acknowledgements

- [INEGI](https://www.inegi.org.mx/) for open census and geographic data
- [SEP](https://www.sep.gob.mx/) for education statistics
- [ANUIES](https://www.anuies.mx/) for higher education enrollment data
