"""
tests/test_model.py
-------------------
Unit tests for the Bayesian probability model.
"""

import numpy as np
import pytest
from src.model import ProdigyProbabilityModel, ModelConfig, sensitivity_analysis


@pytest.fixture
def model():
    return ProdigyProbabilityModel(ModelConfig(n_samples=5_000, random_seed=0))


def test_base_rate_samples_shape(model):
    samples = model.sample_base_rate()
    assert samples.shape == (5_000,)


def test_base_rate_samples_in_unit_interval(model):
    samples = model.sample_base_rate()
    assert np.all(samples >= 0) and np.all(samples <= 1)


def test_municipality_estimate_returns_valid_prob(model):
    est = model.estimate_municipality(
        "TestMun", population=100_000, area_km2=50.0,
        is_urban=True, is_high_edu_state=False, marginalization_index=0.1
    )
    assert 0 <= est.prob_at_least_one <= 1
    assert est.expected_count >= 0
    assert est.credible_interval_95[0] <= est.credible_interval_95[1]


def test_higher_population_yields_higher_prob(model):
    est_small = model.estimate_municipality("Small", 10_000, 10.0)
    est_large = model.estimate_municipality("Large", 1_000_000, 100.0)
    assert est_large.prob_at_least_one > est_small.prob_at_least_one


def test_radius_estimate_scales_with_area(model):
    mun = model.estimate_municipality("TestMun", 500_000, 100.0, is_urban=True)
    r1 = model.estimate_radius(mun, radius_km=1.0)
    r5 = model.estimate_radius(mun, radius_km=5.0)
    assert r5.expected_count > r1.expected_count
    assert r5.prob_at_least_one > r1.prob_at_least_one


def test_radius_exceeding_municipality_capped(model):
    mun = model.estimate_municipality("Small", 10_000, 1.0, is_urban=True)
    # Radius much larger than municipality
    r = model.estimate_radius(mun, radius_km=100.0)
    # Area ratio should be capped at 1.0
    assert r.expected_count <= mun.expected_count * 1.001  # tiny float tolerance


def test_high_edu_state_multiplier(model):
    est_normal = model.estimate_municipality("Normal", 100_000, 50.0, is_high_edu_state=False)
    est_high = model.estimate_municipality("HighEdu", 100_000, 50.0, is_high_edu_state=True)
    assert est_high.expected_count > est_normal.expected_count


def test_marginalization_reduces_probability(model):
    est_low = model.estimate_municipality("LowMarg", 100_000, 50.0, marginalization_index=0.0)
    est_high = model.estimate_municipality("HighMarg", 100_000, 50.0, marginalization_index=0.9)
    assert est_low.prob_at_least_one > est_high.prob_at_least_one


def test_sensitivity_analysis_shape():
    base_rates = [1e-7, 1e-6, 1e-5]
    populations = [10_000, 100_000]
    df = sensitivity_analysis(base_rates, populations)
    assert len(df) == len(base_rates) * len(populations)
    assert "prob_at_least_one" in df.columns


def test_batch_estimate(model):
    import pandas as pd
    muns = pd.DataFrame([
        {"municipality_name": "A", "total_population": 50000, "area_km2": 30.0,
         "is_urban": True, "is_high_edu_state": False, "marginalization_index": 0.1},
        {"municipality_name": "B", "total_population": 200000, "area_km2": 80.0,
         "is_urban": True, "is_high_edu_state": True, "marginalization_index": 0.05},
    ])
    results = model.estimate_batch(muns)
    assert len(results) == 2
    assert all(0 <= p <= 1 for p in results["prob_at_least_one"])
