# Customer Retention & Experimentation Platform

A notebook-first portfolio project for a fictional streaming service retaining free-trial users around the 2026 World Cup. **Only milestone 1 is implemented:** synthetic customer behavior, randomized offers, conversion, day-90 retention, offer costs, and net revenue. No real customer data or external services are required.

## Start here

Open [`notebooks/01_synthetic_retention_experiment.ipynb`](notebooks/01_synthetic_retention_experiment.ipynb). It explains the experiment, every synthetic assumption, the outputs, and their limitations. Run from top to bottom.

With Python 3.11 or newer, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

For Google Colab, open the notebook, clone this repository with its actual GitHub URL in a setup cell, change to the cloned directory using `%cd`, and run `%pip install -r requirements.txt`. Alternatively upload the notebook and `retention/` together. See notebook setup instructions. Dependencies use supported version ranges; generated run metadata records actual NumPy/pandas versions.

## Files and data boundaries

- `notebooks/01_synthetic_retention_experiment.ipynb`: primary walkthrough with executed previews, descriptive tables, charts, and integrity checks.
- `retention/simulation.py`: reusable seeded simulator, explicit pre-offer feature allowlist, and validation.
- `tests/test_simulation.py`: reproducibility, invariants, randomization, and oracle calibration checks.
- `data/generated/customers.csv`: generated observed data, ignored by Git.
- `data/evaluation_only/oracle_probabilities.csv`: separate four-offer probability answer key, ignored by Git; never a model feature source.
- `data/generated/run_metadata.json`: seed, sample size, feature allowlist, and package versions, ignored by Git.

The default run creates 12,000 eligible trial customers with independent 25% assignments to control, a 20% discount, a 7-day extension, or a plan recommendation. Everyone is followed 90 days from assignment. Revenue is prorated over paid days and subtracts incremental offer costs. It is not profit. All offer effects and costs are invented assumptions. Descriptive arm comparisons in this milestone do not establish statistical significance.

## Validate and execute

```bash
python -m pytest -q
python - <<'PY'
import nbformat
from nbclient import NotebookClient
from pathlib import Path
path = Path('notebooks/01_synthetic_retention_experiment.ipynb')
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute()
nbformat.write(nb, path)
PY
```

`data/` is wholly ignored, including answer keys and any large regenerated files. Notebook outputs are limited to small previews and aggregates. Do not force-add generated data.

## Repository redesign

The previous CRM notebook referenced a different Parquet path and columns than its generator produced. The old generator simulated raw CRM formatting problems rather than randomized offer outcomes; its validation also assumed a fixed 100,000-customer run. Those files and the old CRM sample were replaced. ClickHouse Docker configuration, ingestion scaffolding, and empty dbt/SQL/config placeholders were removed because they do not serve this milestone. The useful ideas retained are a fictional streaming setting, World Cup dates, seeded generation, validation, and ignoring generated data.

Later milestones will analyze the randomized experiment, compare offers with appropriate A/B testing, estimate customer-specific offer benefits with ML, and add dbt transformations. None is built here.
