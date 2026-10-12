# Customer Retention & Experimentation Platform

**After the World Cup brings in free-trial customers, which action improves paid retention enough to be worth its cost?**

This is a notebook-first learning project for a fictional streaming service. All customers, behavior, offer effects, and costs are simulated. Results do not show how real customers behave.

## Which notebook should I run first?

**Run Notebook 1 to create the data, then Notebook 2 to analyze it.** Run each notebook from top to bottom, in the same repository checkout.

1. [Notebook 1: create the experiment](notebooks/01_synthetic_retention_experiment.ipynb) explains the scenario, creates 12,000 fictional customers, previews one customer row, checks the data, and saves it.
2. [Notebook 2: analyze the saved data](notebooks/02_experiment_analysis.ipynb) reads that file, shows group results, compares retention and revenue with uncertainty ranges, and ends with a separate simulation answer-key check. It does **not** generate replacement customers. Missing files give an instruction to run Notebook 1 first.

Each section starts with a question, explains the next code cell, and interprets the output. Optional appendices explain the statistical names and formulas.

## How do I start locally?

Use Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open Notebook 1, run all its cells, then open Notebook 2 and run all its cells. Jupyter can start from the repository root or the `notebooks/` folder.

## How do I use Google Colab?

Clone the branch in a Colab setup cell:

```python
!git clone --branch milestone-1-synthetic-experiment https://github.com/KDHAN123/Customer-decisioning-platform.git
%cd /content/Customer-decisioning-platform
%pip install -r requirements.txt
```

Run Notebook 1 to create the data, then Notebook 2 to analyze it. Use the same checkout and data folder. Separate Colab notebooks can have separate runtimes: copy Notebook 1's generated `data/` files to the second checkout, or run Notebook 1 there before Notebook 2. Uploading only a notebook does not include the `retention/` Python helpers or generated files.

## Where do the files go?

| File or folder | Who writes it? | Who reads it? |
|---|---|---|
| `data/generated/customers.csv` | Notebook 1 | Notebook 2's main analysis |
| `data/generated/run_metadata.json` | Notebook 1 | A record of the seed and simulation settings |
| `data/evaluation_only/oracle_probabilities.csv` | Notebook 1 | Notebook 2's final simulation check only |
| `data/milestone2/*comparisons.csv` | Notebook 2 | Full-precision retention, conversion, and revenue results |
| `data/milestone2/analysis_run_metadata.json` | Notebook 2 | A record of the input file's fingerprint and analysis settings |
| `data/milestone2/oracle_estimate_diagnostic.csv` | Notebook 2 | A separate final comparison with the invented answer key |

Everything under `data/` is ignored by Git. Notebook outputs contain small previews, group tables, and charts, not the full datasets. Rerunning Notebook 1 overwrites the inputs; rerun Notebook 2 afterward so its results match. Older local exports may still exist under `data/milestone2/generated/`, but the current notebooks do not read them.

## What experiment are we studying?

Eligible customers are still viewing-active and have not canceled on trial day 13, one day before a scheduled 14-day trial ends. Each has a one-in-three chance of **no special offer**, a **recommendation for relevant non-World-Cup content**, or a **20% discount on the first paid month**.

The main outcome is still paying 90 days after scheduled trial end. All assigned customers count, including nonconverters. Conversion by trial end is a supporting outcome. Subscription revenue already subtracts discounts; the second revenue measure also subtracts recommendation delivery cost. Neither is profit because other operating costs are missing.

Notebook 2 keeps the existing statistical calculations: it adjusts for two main retention comparisons and a separate set of six supporting conversion/revenue comparisons. A higher sample average does not automatically make an offer the winner. The simulated effects vary across customers and are planted assumptions, not discoveries. Only behavior before the offer could be used as future model features; oracle probabilities stay separate.

## Where can I find the details?

- [Simulation assumptions](docs/assumptions.md): population, timeline, behavior, planted effects, and money definitions.
- [Analysis details](docs/experiment_analysis.md): tests, confidence intervals, comparison adjustments, and limits.
- `retention/simulation.py` and `retention/analysis.py`: the existing generator and statistical calculations.
- `retention/data.py`: CSV readers with friendly missing-file instructions.
- `retention/reporting.py`: readable comparison tables, charts, and the separate answer-key check.
- `tests/`: simulation, statistical-reference, and data-handoff checks.

Milestones 1 and 2 are implemented. ML targeting, uplift modeling, and dbt remain later work.

## How do I check and execute everything?

```bash
python -m pytest -q
python - <<'PY'
import nbformat
from nbclient import NotebookClient
from pathlib import Path
for filename in ['01_synthetic_retention_experiment.ipynb', '02_experiment_analysis.ipynb']:
    path = Path('notebooks') / filename
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=180, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute()
    nbformat.write(notebook, path)
PY
git diff --check
```
