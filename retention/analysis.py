"""Observed-data experiment comparisons; never consumes simulation oracle values."""
from numbers import Integral

import numpy as np
import pandas as pd
from scipy import stats

from retention.simulation import ARMS


def _groups(customers, outcome, alpha, family_size):
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    if isinstance(family_size, bool) or not isinstance(family_size, Integral) or family_size < 2:
        raise ValueError('family_size must cover at least the two offer comparisons')
    if not customers.customer_id.is_unique or customers.customer_id.isna().any():
        raise ValueError('One complete row per randomized customer is required')
    if customers.assigned_offer.isna().any() or set(customers.assigned_offer) != set(ARMS):
        raise ValueError('All three randomized groups must be present; no unknown groups')
    # Only select the requested observed outcome. No automatic feature selection.
    if outcome.startswith(('p_', 'effect_')):
        raise ValueError('Oracle probabilities and effects are evaluation-only')
    values = customers[outcome]
    if not pd.api.types.is_numeric_dtype(values) or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('Outcome must be complete and finite; do not drop randomized rows')
    groups = {arm: values[customers.assigned_offer.eq(arm)].to_numpy(dtype=float) for arm in ARMS}
    if any(len(x) < 2 for x in groups.values()):
        raise ValueError('Each randomized arm needs at least two observations')
    return groups


def compare_binary(customers, outcome='retained_day90', alpha=.05, family_size=2):
    """Risk differences (fractions), unpooled Wald CIs, pooled two-proportion z tests.

    family_size=2 protects the primary family. Pass 6 for the exploratory family
    (two offers times conversion and two revenue outcomes). Bonferroni protects
    families even with a shared control. Requires >=10 successes and failures in
    each arm; these normal approximations are not suitable for sparse outcomes.
    """
    groups = _groups(customers, outcome, alpha, family_size)
    if any(not np.isin(x, [0, 1]).all() for x in groups.values()):
        raise ValueError('Binary outcome must contain only zero/one or booleans')
    if any(min(x.sum(), len(x) - x.sum()) < 10 for x in groups.values()):
        raise ValueError('Normal approximation requires >=10 successes and failures per arm')
    control = groups['control']
    n0, p0 = len(control), control.mean()
    rows = []
    for offer in ARMS[1:]:
        x = groups[offer]
        n1, p1 = len(x), x.mean()
        effect = p1 - p0
        se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
        pooled = (x.sum() + control.sum()) / (n1 + n0)
        null_se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n0))
        p_value = 2 * stats.norm.sf(abs(effect / null_se))
        critical = stats.norm.ppf(1 - alpha / 2)
        adjusted_critical = stats.norm.ppf(1 - alpha / (2 * family_size))
        rows.append(dict(
            outcome=outcome, offer=offer, n_control=n0, n_offer=n1,
            control_mean=p0, offer_mean=p1, estimate=effect, standard_error=se,
            ci_low=effect - critical * se, ci_high=effect + critical * se,
            ci_adjusted_low=effect - adjusted_critical * se,
            ci_adjusted_high=effect + adjusted_critical * se,
            p_value=p_value, p_adjusted=min(1., family_size * p_value),
            family_size=family_size, marginal_adjusted_confidence=1 - alpha / family_size,
            test='two-sided pooled two-proportion z',
        ))
    return pd.DataFrame(rows)


def compare_revenue(customers, outcome='subscription_revenue_90', alpha=.05, family_size=6):
    """Per-assigned-customer mean differences and two-sided Welch t inference.

    Zero and negative revenue rows stay in the sample. Variances need not match.
    Bonferroni family_size must include all hypotheses in the planned family.
    Large independent groups justify approximate mean inference despite skew.
    """
    groups = _groups(customers, outcome, alpha, family_size)
    control = groups['control']
    n0, mean0 = len(control), control.mean()
    variance0 = control.var(ddof=1) / n0
    rows = []
    for offer in ARMS[1:]:
        x = groups[offer]
        n1, mean1 = len(x), x.mean()
        variance1 = x.var(ddof=1) / n1
        se = np.sqrt(variance0 + variance1)
        if se == 0:
            raise ValueError('Welch inference requires a nonzero standard error')
        degrees_freedom = (variance0 + variance1) ** 2 / (
            variance0 ** 2 / (n0 - 1) + variance1 ** 2 / (n1 - 1)
        )
        effect = mean1 - mean0
        p_value = 2 * stats.t.sf(abs(effect / se), degrees_freedom)
        critical = stats.t.ppf(1 - alpha / 2, degrees_freedom)
        adjusted_critical = stats.t.ppf(1 - alpha / (2 * family_size), degrees_freedom)
        rows.append(dict(
            outcome=outcome, offer=offer, n_control=n0, n_offer=n1,
            control_mean=mean0, offer_mean=mean1, estimate=effect, standard_error=se,
            ci_low=effect - critical * se, ci_high=effect + critical * se,
            ci_adjusted_low=effect - adjusted_critical * se,
            ci_adjusted_high=effect + adjusted_critical * se,
            p_value=p_value, p_adjusted=min(1., family_size * p_value),
            family_size=family_size, marginal_adjusted_confidence=1 - alpha / family_size,
            degrees_freedom=degrees_freedom, test='two-sided Welch t',
        ))
    return pd.DataFrame(rows)
