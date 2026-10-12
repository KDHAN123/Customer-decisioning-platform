"""Reproducible data-generating process; no fitted models or inference."""
from numbers import Integral

import numpy as np
import pandas as pd

ARMS = ('control', 'content_recommendation', 'discount_20pct')
# Explicit allowlist: outcomes, assignment, IDs, dates, and oracle values excluded.
PRE_OFFER_FEATURES = (
    'active_days_pre', 'active_days_last3_pre', 'watch_hours_pre',
    'world_cup_share_pre', 'non_world_cup_hours_pre', 'content_breadth_pre',
    'pricing_page_visits_pre', 'discovery_sessions_pre',
)
MONTHLY_PRICE = 15.0
FOLLOWUP_DAYS = 90
TRIAL_DAYS = 14
OFFER_DAY = 13
RECOMMENDATION_DELIVERY_COST = .10


def simulate_experiment(n_customers=12000, seed=2026):
    """Return observed customer rows and a separate long-format oracle answer key.

    Generate a cohort conditional on eligibility on trial day 13: still active
    (at least one active day in the preceding three days), with no cancellation.
    Behaviors summarize only the 13 completed days before assignment. Each of
    three arms has probability 1/3; follow-up ends 90 days after scheduled trial
    end, not assignment. See docs/assumptions.md for all planted assumptions.
    """
    if isinstance(n_customers, bool) or not isinstance(n_customers, Integral) or n_customers < 1:
        raise ValueError('n_customers must be a positive integer')
    features_rng, assignment_rng, outcome_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    ]
    n = int(n_customers)
    # Latent preferences influence behavior and response; never exported as features.
    engagement = features_rng.beta(2, 2, n)
    world_cup_interest = features_rng.beta(3, 2, n)
    price_sensitivity = features_rng.beta(2, 2, n)
    # Model eligible users directly, rather than silently filtering after randomization.
    # A recent active day is guaranteed because eligibility conditions on activity.
    recent_active = 1 + features_rng.binomial(2, .15 + .7 * engagement)
    active = features_rng.binomial(10, .15 + .7 * engagement) + recent_active
    hours = np.round(active * features_rng.gamma(2, .65, n), 2)
    world_cup_share = np.round(world_cup_interest, 3)
    non_world_cup_hours = np.round(hours * (1 - world_cup_share), 2)
    breadth = np.where(non_world_cup_hours > 0,
                       1 + features_rng.binomial(4, 1 - world_cup_interest), 0)
    pricing = features_rng.poisson(.3 + 2.5 * price_sensitivity)
    discovery = features_rng.binomial(active, 1 - world_cup_interest)
    signup = pd.Timestamp('2026-05-28') + pd.to_timedelta(
        features_rng.integers(0, 39, n), unit='D'
    )
    assignment = assignment_rng.integers(0, len(ARMS), n)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    base_conversion_logit = -1.5 + .15 * active + .025 * hours - .9 * price_sensitivity
    base_retention_logit = -.2 + .12 * active - 1.3 * world_cup_interest
    # Signed planted effects: recommendations suit World-Cup-heavy users;
    # discounts suit the more price-sensitive. Some users are harmed or indifferent.
    conversion_lifts = np.column_stack([
        np.zeros(n), 1.2 * (world_cup_interest - .55),
        1.5 * (price_sensitivity - .60),
    ])
    retention_lifts = np.column_stack([
        np.zeros(n), 2.2 * (world_cup_interest - .55),
        .7 * (price_sensitivity - .60),
    ])
    p_conversion = sigmoid(base_conversion_logit[:, None] + conversion_lifts)
    p_retention_given_conversion = sigmoid(base_retention_logit[:, None] + retention_lifts)
    # Constant daily churn hazard gives conditional survival at trial-end + 90.
    daily_hazard = 1 - p_retention_given_conversion ** (1 / FOLLOWUP_DAYS)
    rows = np.arange(n)
    converted = outcome_rng.random(n) < p_conversion[rows, assignment]
    # Geometric draw: number of billed daily intervals up to cancellation.
    time_to_churn = outcome_rng.geometric(daily_hazard[rows, assignment])
    horizon = FOLLOWUP_DAYS
    retained = converted & (time_to_churn > horizon)
    paid_days = np.where(converted, np.minimum(time_to_churn, horizon), 0)
    daily_price = MONTHLY_PRICE / 30
    gross = daily_price * paid_days
    discount_cost = np.where(assignment == 2, .2 * daily_price * np.minimum(paid_days, 30), 0)
    delivery_cost = np.where(assignment == 1, RECOMMENDATION_DELIVERY_COST, 0)
    offer_cost = discount_cost + delivery_cost
    customers = pd.DataFrame({
        'customer_id': [f'customer_{i:07d}' for i in range(1, n + 1)],
        'signup_date': signup,
        'feature_window_end_date': signup + pd.Timedelta(days=OFFER_DAY),
        'assignment_date': signup + pd.Timedelta(days=OFFER_DAY),
        'scheduled_trial_end_date': signup + pd.Timedelta(days=TRIAL_DAYS),
        'active_at_offer': True,
        'canceled_before_offer': False,
        'eligible_at_offer': True,
        'followup_end_date': signup + pd.Timedelta(days=TRIAL_DAYS + FOLLOWUP_DAYS),
        'active_days_pre': active,
        'active_days_last3_pre': recent_active,
        'watch_hours_pre': hours,
        'world_cup_share_pre': world_cup_share,
        'non_world_cup_hours_pre': non_world_cup_hours,
        'content_breadth_pre': breadth,
        'pricing_page_visits_pre': pricing,
        'discovery_sessions_pre': discovery,
        'assigned_offer': np.array(ARMS)[assignment],
        'assignment_probability': 1 / len(ARMS),
        'converted': converted,
        'retained_day90': retained,
        'conversion_day': np.where(converted, 0, np.nan),
        'paid_days_90': paid_days,
        'reference_revenue_90': gross,
        'discount_cost_90': discount_cost,
        'delivery_cost_90': delivery_cost,
        'offer_cost_90': offer_cost,
        'subscription_revenue_90': gross - discount_cost,
        'revenue_less_delivery_cost_90': gross - offer_cost,
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
    assert customers.assignment_probability.eq(1 / len(ARMS)).all()
    assert set(PRE_OFFER_FEATURES).issubset(customers.columns)
    assert not any(c.startswith(('p_', 'effect_')) for c in customers.columns)
    assert (customers.assignment_date - customers.signup_date).dt.days.eq(OFFER_DAY).all()
    assert customers.feature_window_end_date.eq(customers.assignment_date).all()
    assert (customers.scheduled_trial_end_date - customers.signup_date).dt.days.eq(TRIAL_DAYS).all()
    assert (customers.followup_end_date - customers.scheduled_trial_end_date).dt.days.eq(FOLLOWUP_DAYS).all()
    assert (customers.assignment_date < customers.scheduled_trial_end_date).all()
    assert customers.signup_date.between('2026-05-28', '2026-07-05').all()
    assert customers.active_at_offer.all() and not customers.canceled_before_offer.any()
    assert customers.eligible_at_offer.all()
    assert customers.active_days_last3_pre.between(1, 3).all()
    assert customers.active_days_pre.between(customers.active_days_last3_pre, OFFER_DAY).all()
    assert customers.world_cup_share_pre.between(0, 1).all()
    assert customers.watch_hours_pre.ge(0).all()
    assert np.allclose(customers.non_world_cup_hours_pre,
                       np.round(customers.watch_hours_pre * (1 - customers.world_cup_share_pre), 2))
    assert customers.content_breadth_pre.between(0, 5).all()
    assert customers.content_breadth_pre.eq(0).eq(customers.non_world_cup_hours_pre.eq(0)).all()
    assert customers.pricing_page_visits_pre.ge(0).all()
    assert customers.discovery_sessions_pre.between(0, customers.active_days_pre).all()
    assert (~customers.retained_day90 | customers.converted).all()
    assert customers.conversion_day.isna().eq(~customers.converted).all()
    assert customers.loc[customers.converted, 'conversion_day'].eq(0).all()
    assert customers.paid_days_90.between(0, FOLLOWUP_DAYS).all()
    assert customers.loc[~customers.converted, 'paid_days_90'].eq(0).all()
    assert customers.loc[customers.retained_day90, 'paid_days_90'].eq(FOLLOWUP_DAYS).all()
    expected_discount = np.where(customers.assigned_offer.eq('discount_20pct'),
                                 .2 * MONTHLY_PRICE / 30 * customers.paid_days_90.clip(upper=30), 0)
    expected_delivery = np.where(customers.assigned_offer.eq('content_recommendation'),
                                 RECOMMENDATION_DELIVERY_COST, 0)
    assert np.allclose(customers.discount_cost_90, expected_discount)
    assert np.allclose(customers.delivery_cost_90, expected_delivery)
    assert np.allclose(customers.reference_revenue_90, customers.paid_days_90 * MONTHLY_PRICE / 30)
    assert np.allclose(customers.subscription_revenue_90,
                       customers.reference_revenue_90 - customers.discount_cost_90)
    assert np.allclose(customers.offer_cost_90, customers.discount_cost_90 + customers.delivery_cost_90)
    assert np.allclose(customers.revenue_less_delivery_cost_90,
                       customers.subscription_revenue_90 - customers.delivery_cost_90)
    assert len(oracle) == len(customers) * len(ARMS)
    assert not oracle.duplicated(['customer_id', 'offer']).any()
    assert set(oracle.customer_id) == set(customers.customer_id)
    assert oracle.groupby('customer_id').offer.agg(set).apply(lambda x: x == set(ARMS)).all()
    probability_columns = [c for c in oracle if c.startswith('p_')]
    assert oracle[probability_columns].notna().all().all()
    assert oracle[probability_columns].ge(0).all().all() and oracle[probability_columns].le(1).all().all()
    assert np.allclose(oracle.p_retained_day90,
                       oracle.p_conversion * oracle.p_retained_day90_given_conversion)
