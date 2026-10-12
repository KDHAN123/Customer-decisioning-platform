import numpy as np
import pandas as pd
import pytest

from retention.analysis import compare_binary, compare_revenue
from retention.data import load_customers, load_oracle
from retention.reporting import comparison_table, oracle_comparison
from retention.simulation import simulate_experiment, validate_experiment


def test_csv_handoff_preserves_rows_dates_and_statistical_results(tmp_path):
    customers, oracle = simulate_experiment()
    customer_path = tmp_path / 'customers.csv'
    oracle_path = tmp_path / 'oracle.csv'
    customers.to_csv(customer_path, index=False)
    oracle.to_csv(oracle_path, index=False)
    # Loading observed data must work even when there is no oracle file.
    oracle_path.unlink()
    loaded = load_customers(customer_path)
    pd.testing.assert_frame_equal(customers, loaded, check_exact=False)
    for outcome in ['retained_day90', 'converted']:
        pd.testing.assert_frame_equal(compare_binary(customers, outcome), compare_binary(loaded, outcome))
    for outcome in ['subscription_revenue_90', 'revenue_less_delivery_cost_90']:
        pd.testing.assert_frame_equal(compare_revenue(customers, outcome), compare_revenue(loaded, outcome))
    oracle.to_csv(oracle_path, index=False)
    validate_experiment(loaded, load_oracle(oracle_path))


@pytest.mark.parametrize('loader', [load_customers, load_oracle])
def test_missing_file_explains_notebook_order(tmp_path, loader):
    with pytest.raises(FileNotFoundError, match='Run Notebook 1'):
        loader(tmp_path / 'missing.csv')


def test_display_conversion_and_oracle_diagnostic_keep_calculations_correct():
    customers, oracle = simulate_experiment()
    result = compare_binary(customers)
    before = result.copy()
    shown = comparison_table(result, percentage_points=True)
    assert np.allclose(shown['Difference (pp)'], np.round(100 * result.estimate, 3))
    check = oracle_comparison(result, oracle, 'p_retained_day90')
    means = oracle.groupby('offer').p_retained_day90.mean()
    for row in check.itertuples():
        assert row.planted_cohort_effect_pp == pytest.approx(100 * (means[row.offer] - means['control']))
    pd.testing.assert_frame_equal(result, before)
