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
    assert (abs(counts - 40000 / 3) < 5 * np.sqrt(40000 * (1 / 3) * (2 / 3))).all()
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
    broken.loc[0, 'subscription_revenue_90'] += 1
    with pytest.raises(AssertionError):
        validate_experiment(broken, oracle)
    broken = customers.copy()
    broken['p_conversion'] = .5
    with pytest.raises(AssertionError):
        validate_experiment(broken, oracle)


def test_eligibility_and_trial_end_clock():
    customers, _ = simulate_experiment(1000)
    assert customers.eligible_at_offer.all()
    assert customers.active_days_last3_pre.ge(1).all()
    assert not customers.canceled_before_offer.any()
    assert (customers.scheduled_trial_end_date - customers.assignment_date).dt.days.eq(1).all()
    assert (customers.followup_end_date - customers.assignment_date).dt.days.eq(91).all()
    assert (customers.followup_end_date - customers.scheduled_trial_end_date).dt.days.eq(90).all()
    assert customers.feature_window_end_date.eq(customers.assignment_date).all()
    assert not {'active_at_offer', 'canceled_before_offer', 'eligible_at_offer'} & set(PRE_OFFER_FEATURES)


def test_offers_can_help_or_harm_in_simulation_ground_truth():
    customers, oracle = simulate_experiment(12000)
    assert set(customers.assigned_offer) == {'control', 'content_recommendation', 'discount_20pct'}
    for outcome in ['p_conversion', 'p_retained_day90']:
        probabilities = oracle.pivot(index='customer_id', columns='offer', values=outcome)
        for arm in ['content_recommendation', 'discount_20pct']:
            effects = probabilities[arm] - probabilities['control']
            assert effects.gt(0).any() and effects.lt(0).any()
    probabilities = oracle.pivot(index='customer_id', columns='offer', values='p_retained_day90')
    effects = probabilities['content_recommendation'] - probabilities['control']
    world_cup_share = customers.set_index('customer_id').world_cup_share_pre
    assert effects[world_cup_share > .551].gt(0).all()
    assert effects[world_cup_share < .549].lt(0).all()


def test_discount_not_double_counted_and_nonconverter_delivery_cost():
    customers, oracle = simulate_experiment(2000)
    validate_experiment(customers, oracle)
    assert np.allclose(customers.subscription_revenue_90,
                       customers.reference_revenue_90 - customers.discount_cost_90)
    assert np.allclose(customers.revenue_less_delivery_cost_90,
                       customers.subscription_revenue_90 - customers.delivery_cost_90)
    assert customers.discount_cost_90.between(0, 3).all()
    nonconverters = customers.loc[~customers.converted]
    assert nonconverters.subscription_revenue_90.eq(0).all()
    recommended = nonconverters.loc[nonconverters.assigned_offer.eq('content_recommendation')]
    assert len(recommended) > 0
    assert recommended.revenue_less_delivery_cost_90.eq(-.1).all()


@pytest.mark.parametrize('column, value', [
    ('canceled_before_offer', True), ('active_days_last3_pre', 0),
    ('active_days_pre', 14), ('eligible_at_offer', False),
])
def test_validation_rejects_ineligible_or_post_offer_behavior(column, value):
    customers, oracle = simulate_experiment(100)
    customers.loc[0, column] = value
    with pytest.raises(AssertionError):
        validate_experiment(customers, oracle)
