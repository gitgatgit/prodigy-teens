"""
model.py
--------
Bayesian Poisson model for estimating the probability of extreme academic
acceleration (graduate degree at age ≤ 17) within arbitrary geographic regions.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & Priors
# ---------------------------------------------------------------------------

# Base rate estimates from literature review and SEP/ANUIES data analysis
# P(graduate degree | age <= 17) nationally — conservative estimate
NATIONAL_BASE_RATE_PRIOR_ALPHA = 2.0    # shape parameter (Beta prior)
NATIONAL_BASE_RATE_PRIOR_BETA = 50000.0  # => mean ~0.00004 (4 per 100,000)

# Mexico 2022 graduate completions (master's + doctorate), approx from ANUIES
TOTAL_GRADUATE_COMPLETIONS_MX_2022 = 160_000

# Mexico total population 2020 (INEGI Census)
MEXICO_TOTAL_POPULATION = 126_014_024


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RegionEstimate:
    """Probability estimate for a geographic region."""
    region_name: str
    population: int
    expected_count: float
    prob_at_least_one: float
    credible_interval_95: tuple[float, float]
    mean_count: float
    std_count: float
    area_km2: Optional[float] = None
    radius_km: Optional[float] = None


@dataclass
class ModelConfig:
    """Configuration for the probability model."""
    base_rate_alpha: float = NATIONAL_BASE_RATE_PRIOR_ALPHA
    base_rate_beta: float = NATIONAL_BASE_RATE_PRIOR_BETA
    n_samples: int = 10_000
    random_seed: int = 42
    # Adjustment factors (multiplicative)
    urban_multiplier: float = 1.8       # Urban areas have more university access
    high_edu_state_multiplier: float = 1.5  # States with top universities (CDMX, NL, JAL)
    poverty_discount: float = 0.6       # High marginalization reduces probability


# ---------------------------------------------------------------------------
# Core Model
# ---------------------------------------------------------------------------

class ProdigyProbabilityModel:
    """
    Bayesian Poisson model for estimating prodigy density.

    The generative model is:
        base_rate ~ Beta(alpha, beta)
        lambda_municipality = base_rate * population * adjustment_factor
        count ~ Poisson(lambda_municipality)
        P(count >= 1) = 1 - exp(-lambda)

    For radius queries:
        lambda_radius = lambda_municipality * (area_circle / area_municipality)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self._base_rate_samples: Optional[np.ndarray] = None

    def sample_base_rate(self) -> np.ndarray:
        """Draw posterior samples of the national base rate."""
        if self._base_rate_samples is None:
            self._base_rate_samples = self.rng.beta(
                self.config.base_rate_alpha,
                self.config.base_rate_beta,
                size=self.config.n_samples,
            )
        return self._base_rate_samples

    def compute_adjustment_factor(
        self,
        is_urban: bool = True,
        is_high_edu_state: bool = False,
        marginalization_index: float = 0.0,  # 0=low, 1=high
    ) -> float:
        """
        Compute a multiplicative adjustment to the base rate.

        Parameters
        ----------
        is_urban : whether the municipality is classified urban by INEGI
        is_high_edu_state : whether the state hosts major research universities
        marginalization_index : CONAPO marginalization index [0, 1]
        """
        factor = 1.0
        if is_urban:
            factor *= self.config.urban_multiplier
        if is_high_edu_state:
            factor *= self.config.high_edu_state_multiplier
        # Poverty discount: linearly interpolate
        factor *= (1 - marginalization_index * (1 - self.config.poverty_discount))
        return factor

    def estimate_municipality(
        self,
        municipality_name: str,
        population: int,
        area_km2: float,
        is_urban: bool = True,
        is_high_edu_state: bool = False,
        marginalization_index: float = 0.0,
    ) -> RegionEstimate:
        """
        Estimate expected prodigy count and probability for a municipality.
        """
        adj = self.compute_adjustment_factor(is_urban, is_high_edu_state, marginalization_index)
        base_rates = self.sample_base_rate()
        lambdas = base_rates * population * adj

        count_samples = self.rng.poisson(lambdas)
        prob_at_least_one = np.mean(count_samples >= 1)
        ci_low, ci_high = np.percentile(lambdas, [2.5, 97.5])

        return RegionEstimate(
            region_name=municipality_name,
            population=population,
            expected_count=float(np.mean(lambdas)),
            prob_at_least_one=float(prob_at_least_one),
            credible_interval_95=(float(ci_low), float(ci_high)),
            mean_count=float(np.mean(count_samples)),
            std_count=float(np.std(count_samples)),
            area_km2=area_km2,
        )

    def estimate_radius(
        self,
        municipality_estimate: RegionEstimate,
        radius_km: float,
    ) -> RegionEstimate:
        """
        Estimate probability within a circular radius, given a municipality estimate.

        Uses area ratio to scale the municipal lambda.
        Assumes uniform population distribution within the municipality (conservative).
        """
        if municipality_estimate.area_km2 is None or municipality_estimate.area_km2 == 0:
            raise ValueError("Municipality area_km2 must be set and > 0")

        circle_area = np.pi * radius_km ** 2
        area_ratio = min(circle_area / municipality_estimate.area_km2, 1.0)

        # Scale expected count by area ratio
        lambda_radius = municipality_estimate.expected_count * area_ratio
        ci_low = municipality_estimate.credible_interval_95[0] * area_ratio
        ci_high = municipality_estimate.credible_interval_95[1] * area_ratio

        # Poisson probability of at least 1
        prob = 1 - np.exp(-lambda_radius)

        return RegionEstimate(
            region_name=f"{municipality_estimate.region_name} ({radius_km:.1f}km radius)",
            population=int(municipality_estimate.population * area_ratio),
            expected_count=lambda_radius,
            prob_at_least_one=prob,
            credible_interval_95=(ci_low, ci_high),
            mean_count=lambda_radius,
            std_count=np.sqrt(lambda_radius),  # Poisson: var = mean
            area_km2=circle_area,
            radius_km=radius_km,
        )

    def estimate_batch(self, municipalities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate probabilities for all municipalities in a DataFrame.

        Expected columns:
            municipality_name, total_population, area_km2,
            is_urban (bool), is_high_edu_state (bool), marginalization_index (float)
        """
        results = []
        for _, row in municipalities_df.iterrows():
            est = self.estimate_municipality(
                municipality_name=row.get("municipality_name", "Unknown"),
                population=int(row.get("total_population", 0)),
                area_km2=float(row.get("area_km2", 100.0)),
                is_urban=bool(row.get("is_urban", True)),
                is_high_edu_state=bool(row.get("is_high_edu_state", False)),
                marginalization_index=float(row.get("marginalization_index", 0.0)),
            )
            results.append({
                "municipality_name": est.region_name,
                "population": est.population,
                "expected_count": est.expected_count,
                "prob_at_least_one": est.prob_at_least_one,
                "ci_95_low": est.credible_interval_95[0],
                "ci_95_high": est.credible_interval_95[1],
                "area_km2": est.area_km2,
            })

        return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(base_rates: list[float], populations: list[int]) -> pd.DataFrame:
    """
    Grid search over base rate assumptions and population sizes.
    Returns DataFrame of (base_rate, population, prob_at_least_one).
    """
    records = []
    for rate in base_rates:
        for pop in populations:
            lam = rate * pop
            prob = 1 - np.exp(-lam)
            records.append({
                "base_rate_per_million": rate * 1e6,
                "population": pop,
                "expected_count": lam,
                "prob_at_least_one": prob,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = ProdigyProbabilityModel()

    # Benito Juárez (Polanco area) — CDMX municipality
    mun = model.estimate_municipality(
        municipality_name="Benito Juárez, CDMX",
        population=434_153,
        area_km2=26.63,
        is_urban=True,
        is_high_edu_state=True,
        marginalization_index=0.05,
    )
    print(f"\n=== {mun.region_name} ===")
    print(f"  Expected count:      {mun.expected_count:.4f}")
    print(f"  P(at least 1):       {mun.prob_at_least_one:.6f}")
    print(f"  95% CI on lambda:    ({mun.credible_interval_95[0]:.5f}, {mun.credible_interval_95[1]:.5f})")

    radius = model.estimate_radius(mun, radius_km=1.0)
    print(f"\n=== {radius.region_name} ===")
    print(f"  Expected count:      {radius.expected_count:.6f}")
    print(f"  P(at least 1):       {radius.prob_at_least_one:.8f}")
    print(f"  Area:                {radius.area_km2:.2f} km²")
