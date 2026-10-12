"""Readable tables and plots for notebooks; statistical calculations stay unchanged."""
import matplotlib.pyplot as plt
import numpy as np

OFFER_NAMES = {
    'control': 'No special offer',
    'content_recommendation': 'Content recommendation',
    'discount_20pct': '20% discount',
}


def comparison_table(results, percentage_points=False):
    """Show adjusted results with readable headings; retain full precision in CSVs."""
    columns = ['offer', 'n_control', 'n_offer', 'estimate',
               'ci_adjusted_low', 'ci_adjusted_high', 'p_adjusted']
    table = results[columns].copy()
    unit = 'pp' if percentage_points else 'USD per customer'
    if percentage_points:
        table[['estimate', 'ci_adjusted_low', 'ci_adjusted_high']] *= 100
    table['offer'] = table.offer.map(OFFER_NAMES)
    table[['estimate', 'ci_adjusted_low', 'ci_adjusted_high']] = table[
        ['estimate', 'ci_adjusted_low', 'ci_adjusted_high']].round(3)
    table['p_adjusted'] = table.p_adjusted.map(lambda value: f'{value:.4g}')
    return table.rename(columns={
        'offer': 'Offer', 'n_control': 'Control customers', 'n_offer': 'Offer customers',
        'estimate': f'Difference ({unit})', 'ci_adjusted_low': f'Range low ({unit})',
        'ci_adjusted_high': f'Range high ({unit})', 'p_adjusted': 'Adjusted p-value',
    })


def plot_comparison(results, title, percentage_points=False):
    """Plot the existing adjusted intervals without recalculating any result."""
    scale = 100 if percentage_points else 1
    unit = 'percentage points' if percentage_points else 'USD per assigned customer'
    error = np.vstack([results.estimate - results.ci_adjusted_low,
                       results.ci_adjusted_high - results.estimate]) * scale
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.errorbar(results.estimate * scale, np.arange(len(results)), xerr=error,
                fmt='o', capsize=5, color='#366b93')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_yticks(np.arange(len(results)), results.offer.map(OFFER_NAMES))
    ax.set_xlabel(f'Offer minus no special offer ({unit})')
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def oracle_comparison(results, oracle, probability):
    """Final diagnostic: compare to probabilities over the same entire cohort."""
    means = oracle.groupby('offer')[probability].mean()
    check = results[['outcome', 'offer', 'estimate', 'ci_adjusted_low', 'ci_adjusted_high']].copy()
    check['planted_effect'] = check.offer.map(means) - means['control']
    check['observed_minus_planted'] = check.estimate - check.planted_effect
    check['inside_adjusted_interval'] = check.planted_effect.between(
        check.ci_adjusted_low, check.ci_adjusted_high)
    for column in ['estimate', 'ci_adjusted_low', 'ci_adjusted_high',
                   'planted_effect', 'observed_minus_planted']:
        check[column] *= 100
    return check.rename(columns={
        'estimate': 'observed_effect_pp', 'ci_adjusted_low': 'adjusted_ci_low_pp',
        'ci_adjusted_high': 'adjusted_ci_high_pp', 'planted_effect': 'planted_cohort_effect_pp',
        'observed_minus_planted': 'observed_minus_planted_pp',
        'inside_adjusted_interval': 'planted_inside_adjusted_interval',
    })
