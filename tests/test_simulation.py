import numpy as np
import pandas as pd
import pytest

from retention.simulation import ARMS, PRE_OFFER_FEATURES, simulate_experiment, validate_experiment


def test_repeatability_and_global_rng_independence():
    a, key_a = simulate_experiment(250, 42)
    np.random.seed(99)
    np.random.random(100)
    b, key_b = simulate_experiment(250, 42)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(key_a, key_b)
    c, _ = simulate_experiment(250, 43)
    assert not a.equals(c)


@pytest.mark.parametrize('n', [1, 3, 250, 12000])
def test_integrity_and_evaluation_separation(n):
    customers, oracle = simulate_experiment(n)
    validate_experiment(customers, oracle)
    assert len(customers) == n
    assert list(customers[list(PRE_OFFER_FEATURES)].columns) == list(PRE_OFFER_FEATURES)
    assert not set(PRE_OFFER_FEATURES) & set(oracle.columns)


@pytest.mark.parametrize('n', [0, -1, 2.5, True])
def test_invalid_sample_size(n):
    with pytest.raises(ValueError):
        simulate_experiment(n)


def test_randomization_and_probability_calibration():
    customers, oracle = simulate_experiment(40000, 73)
    counts = customers.assigned_offer.value_counts().reindex(ARMS)
    # Broad five-standard-deviation tolerance; never enforce exact balance.
    assert (abs(counts - 10000) < 5 * np.sqrt(40000 * .25 * .75)).all()
    for feature in PRE_OFFER_FEATURES:
        overall_sd = customers[feature].std()
        deviations = customers.groupby('assigned_offer')[feature].mean() - customers[feature].mean()
        assert (deviations.abs() / overall_sd < .06).all()
    assigned_key = customers[['customer_id', 'assigned_offer', 'converted', 'retained_day90']].merge(
        oracle, left_on=['customer_id', 'assigned_offer'], right_on=['customer_id', 'offer'], validate='one_to_one'
    )
    for arm in ARMS:
        arm_rows = assigned_key[assigned_key.assigned_offer.eq(arm)]
        for observed, probability in [('converted', 'p_conversion'), ('retained_day90', 'p_retained_day90')]:
            p = arm_rows[probability]
            sd = np.sqrt((p * (1 - p)).sum())
            assert abs(arm_rows[observed].sum() - p.sum()) < 5 * sd


def test_validation_catches_broken_accounting_and_leakage():
    customers, oracle = simulate_experiment(100)
    broken = customers.copy()
    broken.loc[0, 'net_revenue_90'] += 1
    with pytest.raises(AssertionError):
        validate_experiment(broken, oracle)
    broken = customers.copy()
    broken['p_conversion'] = .5
    with pytest.raises(AssertionError):
        validate_experiment(broken, oracle)
