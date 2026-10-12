"""Read Notebook 1's saved files without generating replacement data."""
from pathlib import Path

import pandas as pd

DATE_COLUMNS = [
    'signup_date', 'feature_window_end_date', 'assignment_date',
    'scheduled_trial_end_date', 'followup_end_date',
]


def load_customers(path):
    """Restore customer dates from CSV; never drop rows or read the oracle."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            'Customer data is missing. Run Notebook 1 '
            '(01_synthetic_retention_experiment.ipynb) from top to bottom first, '
            'then rerun Notebook 2.'
        )
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def load_oracle(path):
    """Use only in the final simulation check, after observed-data analysis."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            'The simulation answer key is missing. Run Notebook 1 '
            '(01_synthetic_retention_experiment.ipynb) from top to bottom first, '
            'then rerun Notebook 2 with its matching customer data.'
        )
    return pd.read_csv(path)
