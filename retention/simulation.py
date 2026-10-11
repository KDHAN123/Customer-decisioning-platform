"""Reproducible data-generating process; no fitted models or inference."""
from numbers import Integral

import numpy as np
import pandas as pd

ARMS = ('control', 'discount_20pct', 'extension_7days', 'plan_recommendation')
# Explicit allowlist: outcomes, assignment, IDs, dates, and oracle values excluded.
PRE_OFFER_FEATURES = (
    'active_days_pre', 'watch_hours_pre', 'sports_share_pre',
    'pricing_page_visits_pre', 'discovery_sessions_pre',
)
MONTHLY_PRICE = 15.0
FOLLOWUP_DAYS = 90


def simulate_experiment(n_customers=12000, seed=2026):
    """Return observed customer rows and a separate long-format oracle answer key.

    All customers are still on a 14-day trial immediately before assignment.
    Behaviors summarize those 14 days. One independent 25% draw assigns each
    customer an offer; only the assigned arm's outcome is realized.
    """
    if isinstance(n_customers, bool) or not isinstance(n_customers, Integral) or n_customers < 1:
        raise ValueError('n_customers must be a positive integer')
    features_rng, assignment_rng, outcome_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    ]
    n = int(n_customers)
    # Latent preferences influence behavior and response; never exported as features.
    engagement = features_rng.beta(2, 2, n)
    sports_interest = features_rng.beta(3, 2, n)
    price_sensitivity = features_rng.beta(2, 2, n)
    active = features_rng.binomial(14, .15 + .7 * engagement)
    hours = np.round(active * features_rng.gamma(2, .65, n), 2)
    sports_share = np.round(sports_interest, 3)
    pricing = features_rng.poisson(.3 + 2.5 * price_sensitivity)
    discovery = features_rng.binomial(active, 1 - sports_interest)
    signup = pd.Timestamp('2026-05-28') + pd.to_timedelta(
        features_rng.integers(0, 39, n), unit='D'
    )
    assignment = assignment_rng.integers(0, len(ARMS), n)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    base_conversion_logit = -1.5 + .15 * active + .025 * hours - .9 * price_sensitivity
    base_retention_logit = -.2 + .12 * active - 1.3 * sports_interest
    # Log-odds increments deliberately create heterogeneous offer benefits.
    conversion_lifts = np.column_stack([
        np.zeros(n), .15 + .95 * price_sensitivity,
        .1 + .6 * (1 - engagement), .1 + .7 * sports_interest,
    ])
    retention_lifts = np.column_stack([
        np.zeros(n), .1 + .25 * price_sensitivity,
        .05 + .25 * (1 - engagement), .2 + .8 * sports_interest,
    ])
    p_conversion = sigmoid(base_conversion_logit[:, None] + conversion_lifts)
    p_retention_given_conversion = sigmoid(base_retention_logit[:, None] + retention_lifts)
    delays = np.array([0, 0, 7, 0])
    # Constant daily churn hazard gives the requested probability at day 90.
    paid_horizons = FOLLOWUP_DAYS - delays
    daily_hazard = 1 - p_retention_given_conversion ** (1 / paid_horizons)
    rows = np.arange(n)
    converted = outcome_rng.random(n) < p_conversion[rows, assignment]
    # Geometric draw: number of billed daily intervals up to cancellation.
    time_to_churn = outcome_rng.geometric(daily_hazard[rows, assignment])
    horizon = paid_horizons[assignment]
    retained = converted & (time_to_churn > horizon)
    paid_days = np.where(converted, np.minimum(time_to_churn, horizon), 0)
    daily_price = MONTHLY_PRICE / 30
    gross = daily_price * paid_days
    discount_cost = np.where(assignment == 1, .2 * daily_price * np.minimum(paid_days, 30), 0)
    delivery_cost = np.select([assignment == 2, assignment == 3], [7 * .15, .10], default=0)
    offer_cost = discount_cost + delivery_cost
    customers = pd.DataFrame({
        'customer_id': [f'customer_{i:07d}' for i in range(1, n + 1)],
        'signup_date': signup,
        'assignment_date': signup + pd.Timedelta(days=14),
        'followup_end_date': signup + pd.Timedelta(days=14 + FOLLOWUP_DAYS),
        'active_days_pre': active,
        'watch_hours_pre': hours,
        'sports_share_pre': sports_share,
        'pricing_page_visits_pre': pricing,
        'discovery_sessions_pre': discovery,
        'assigned_offer': np.array(ARMS)[assignment],
        'assignment_probability': .25,
        'converted': converted,
        'retained_day90': retained,
        'conversion_day': np.where(converted, delays[assignment], np.nan),
        'paid_days_90': paid_days,
        'gross_revenue_90': gross,
        'discount_cost_90': discount_cost,
        'delivery_cost_90': delivery_cost,
        'offer_cost_90': offer_cost,
        'net_revenue_90': gross - offer_cost,
    })
    oracle = pd.DataFrame({
        'customer_id': np.repeat(customers.customer_id.to_numpy(), len(ARMS)),
        'offer': np.tile(ARMS, n),
        'p_conversion': p_conversion.ravel(),
        'p_retained_day90_given_conversion': p_retention_given_conversion.ravel(),
        'p_retained_day90': (p_conversion * p_retention_given_conversion).ravel(),
    })
    return customers, oracle


def validate_experiment(customers, oracle):
    """Fail loudly on broken timelines, probability separation, or accounting."""
    assert customers.customer_id.is_unique
    assert not customers.isna().drop(columns='conversion_day').any().any()
    assert customers.assigned_offer.isin(ARMS).all()
    assert customers.assignment_probability.eq(.25).all()
    assert set(PRE_OFFER_FEATURES).issubset(customers.columns)
    assert not any(c.startswith('p_') for c in customers.columns)
    assert (customers.assignment_date - customers.signup_date).dt.days.eq(14).all()
    assert (customers.followup_end_date - customers.assignment_date).dt.days.eq(90).all()
    assert customers.signup_date.between('2026-05-28', '2026-07-05').all()
    assert customers.active_days_pre.between(0, 14).all()
    assert customers.sports_share_pre.between(0, 1).all()
    assert customers.watch_hours_pre.ge(0).all()
    assert customers.discovery_sessions_pre.between(0, customers.active_days_pre).all()
    assert (~customers.retained_day90 | customers.converted).all()
    expected_delay = customers.assigned_offer.eq('extension_7days').astype(int) * 7
    assert customers.conversion_day.isna().eq(~customers.converted).all()
    assert customers.loc[customers.converted, 'conversion_day'].eq(expected_delay[customers.converted]).all()
    assert customers.paid_days_90.between(0, 90 - expected_delay).all()
    assert customers.loc[~customers.converted, 'paid_days_90'].eq(0).all()
    assert customers.loc[customers.retained_day90, 'paid_days_90'].eq((90 - expected_delay)[customers.retained_day90]).all()
    expected_discount = np.where(customers.assigned_offer.eq('discount_20pct'),
                                 .2 * MONTHLY_PRICE / 30 * customers.paid_days_90.clip(upper=30), 0)
    expected_delivery = np.select([customers.assigned_offer.eq('extension_7days'),
                                  customers.assigned_offer.eq('plan_recommendation')], [1.05, .1], default=0)
    assert np.allclose(customers.discount_cost_90, expected_discount)
    assert np.allclose(customers.delivery_cost_90, expected_delivery)
    assert np.allclose(customers.gross_revenue_90, customers.paid_days_90 * MONTHLY_PRICE / 30)
    assert np.allclose(customers.offer_cost_90, customers.discount_cost_90 + customers.delivery_cost_90)
    assert np.allclose(customers.net_revenue_90, customers.gross_revenue_90 - customers.offer_cost_90)
    assert len(oracle) == len(customers) * len(ARMS)
    assert not oracle.duplicated(['customer_id', 'offer']).any()
    assert set(oracle.customer_id) == set(customers.customer_id)
    assert oracle.groupby('customer_id').offer.agg(set).apply(lambda x: x == set(ARMS)).all()
    probability_columns = [c for c in oracle if c.startswith('p_')]
    assert oracle[probability_columns].notna().all().all()
    assert oracle[probability_columns].ge(0).all().all() and oracle[probability_columns].le(1).all().all()
    assert np.allclose(oracle.p_retained_day90,
                       oracle.p_conversion * oracle.p_retained_day90_given_conversion)
