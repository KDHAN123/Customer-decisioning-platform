import numpy as np
import pandas as pd
import pytest
from scipy import stats

from retention.analysis import compare_binary, compare_revenue
from retention.simulation import ARMS, simulate_experiment


def binary_fixture():
    # Exactly known arm rates .20, .30, .10; includes all nonconverters.
    parts = []
    for arm, successes in zip(ARMS, [20, 30, 10]):
        parts.append(pd.DataFrame({
            'customer_id': [f'{arm}_{i}' for i in range(100)],
            'assigned_offer': arm,
            'retained_day90': [1] * successes + [0] * (100 - successes),
        }))
    return pd.concat(parts, ignore_index=True)


def test_binary_known_effect_denominators_and_reference_test():
    data = binary_fixture()
    result = compare_binary(data)
    assert result.n_control.eq(100).all() and result.n_offer.eq(100).all()
    assert np.allclose(result.estimate, [.1, -.1])
    row = result.iloc[0]
    se = np.sqrt(.2 * .8 / 100 + .3 * .7 / 100)
    assert row.standard_error == pytest.approx(se)
    assert row.ci_adjusted_low == pytest.approx(.1 - stats.norm.ppf(.9875) * se)
    pooled = .25
    reference_p = 2 * stats.norm.sf(.1 / np.sqrt(pooled * (1 - pooled) * .02))
    assert row.p_value == pytest.approx(reference_p)
    assert row.p_adjusted == pytest.approx(min(1, 2 * reference_p))
    assert (result.ci_adjusted_low <= result.ci_low).all()
    assert (result.ci_adjusted_high >= result.ci_high).all()


def test_welch_matches_scipy_and_retains_zero_negative_rows():
    values = [[0, 0, 0, 3, 9, 15], [-.1, -.1, -.1, 6, 9, 14.9], [0, 0, 0, 1, 2, 3]]
    data = pd.concat([pd.DataFrame({
        'customer_id': [f'{arm}_{i}' for i in range(6)],
        'assigned_offer': arm, 'revenue': x,
    }) for arm, x in zip(ARMS, values)], ignore_index=True)
    result = compare_revenue(data, 'revenue')
    assert result.n_control.eq(6).all() and result.n_offer.eq(6).all()
    for i, x in enumerate(values[1:]):
        reference = stats.ttest_ind(x, values[0], equal_var=False)
        row = result.iloc[i]
        assert row.estimate == pytest.approx(np.mean(x) - np.mean(values[0]))
        assert row.p_value == pytest.approx(reference.pvalue)
        assert row.degrees_freedom == pytest.approx(reference.df)
        assert row.ci_low == pytest.approx(reference.confidence_interval().low)
        assert row.ci_high == pytest.approx(reference.confidence_interval().high)
        assert row.p_adjusted == pytest.approx(min(1, 6 * reference.pvalue))
        assert row.ci_adjusted_high >= row.ci_high


def test_observed_results_do_not_depend_on_oracle_or_row_order():
    customers, _ = simulate_experiment()
    before = customers.copy(deep=True)
    result = compare_binary(customers)
    with_extra = customers.assign(p_retained_day90=999, effect_retention=-999)
    pd.testing.assert_frame_equal(result, compare_binary(with_extra.sample(frac=1, random_state=3)))
    pd.testing.assert_frame_equal(customers, before)
    with pytest.raises(ValueError, match='evaluation-only'):
        compare_binary(with_extra, 'p_retained_day90')


@pytest.mark.parametrize('issue', ['missing', 'duplicate', 'unknown', 'absent_arm', 'sparse', 'nonbinary'])
def test_analysis_rejects_invalid_samples_without_silently_dropping_rows(issue):
    data = binary_fixture()
    if issue == 'missing':
        data.loc[0, 'retained_day90'] = np.nan
    elif issue == 'duplicate':
        data.loc[0, 'customer_id'] = data.loc[1, 'customer_id']
    elif issue == 'unknown':
        data.loc[0, 'assigned_offer'] = 'unknown'
    elif issue == 'absent_arm':
        data = data[data.assigned_offer.ne('discount_20pct')]
    elif issue == 'sparse':
        data['retained_day90'] = 0
    elif issue == 'nonbinary':
        data['retained_day90'] = data.retained_day90.astype(float)
        data.loc[0, 'retained_day90'] = .5
    with pytest.raises(ValueError):
        compare_binary(data)


def test_secondary_family_adjustment_is_consistent():
    data = binary_fixture()
    primary = compare_binary(data, family_size=2)
    secondary = compare_binary(data, family_size=6)
    assert np.allclose(primary.estimate, secondary.estimate)
    assert np.allclose(secondary.p_adjusted, np.minimum(1, secondary.p_value * 6))
    assert (secondary.ci_adjusted_low < primary.ci_adjusted_low).all()
    assert (secondary.ci_adjusted_high > primary.ci_adjusted_high).all()
