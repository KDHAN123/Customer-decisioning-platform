# Customer Retention & Experimentation Platform

**After the World Cup brings in free-trial customers, which action improves paid retention enough to be worth its cost?**

A notebook-first portfolio project for a fictional streaming service. **Milestones 1 and 2 are implemented:** generate the three-action randomized experiment, then analyze average retention, conversion, and revenue differences with uncertainty and multiple-comparison adjustments. All effects are planted assumptions; no real customer data or external services are required.

## Start here

Open [`notebooks/01_synthetic_retention_experiment.ipynb`](notebooks/01_synthetic_retention_experiment.ipynb). It explains the experiment, every synthetic assumption, the outputs, and their limitations. Run from top to bottom. Then open [`notebooks/02_experiment_analysis.ipynb`](notebooks/02_experiment_analysis.ipynb) for primary retention tests, exploratory conversion and revenue comparisons, adjusted confidence intervals, and a final separate oracle diagnostic. Notebook 2 can also run independently: it regenerates the unchanged simulator and preserves Notebook 1 exports.

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
- `notebooks/02_experiment_analysis.ipynb`: randomized comparisons using all assigned customers, with two primary and six exploratory secondary comparisons.
- `retention/analysis.py`: reusable proportion and Welch inference helpers; no oracle inputs or fitted models.
- `tests/test_analysis.py`: known-effect, statistical-reference, denominator, multiplicity, and input-integrity checks.
- [`docs/experiment_analysis.md`](docs/experiment_analysis.md): estimands, tests, adjustment families, uncertainty, and oracle boundary.
- `retention/simulation.py`: reusable seeded simulator, explicit pre-offer feature allowlist, and validation.
- `tests/test_simulation.py`: reproducibility, eligibility, timing, accounting, randomization, varied effects, and oracle calibration checks.
- [`docs/assumptions.md`](docs/assumptions.md): full business scenario, synthetic distributions, planted response formulas, and revenue definitions.
- `data/generated/customers.csv`: generated observed data, ignored by Git.
- `data/evaluation_only/oracle_probabilities.csv`: separate three-action probability answer key, ignored by Git; never a model feature source.
- `data/generated/run_metadata.json`: seed, sample size, feature allowlist, and package versions, ignored by Git.
- `data/milestone2/`: separate observed data, inference tables, run metadata, and evaluation-only oracle artifacts from Notebook 2, all ignored by Git.

The default run creates 12,000 customers conditional on eligibility on **trial day 13**: still active (a viewing-active day in the preceding three completed days) and not canceled. It does not simulate ineligible signups. All potential future ML features describe behavior before the offer, including non-World-Cup viewing and content breadth; audit flags, dates, assignment, outcomes, costs, and oracle probabilities are excluded.

Each customer receives an independent **1/3** assignment to **no special offer**, a **recommendation for relevant non-World-Cup content**, or a **20% discount on the first paid month**. Recommendations and discounts can help some customers and harm others in this invented world. There is no universally helpful action or guaranteed average winner.

**Primary outcome:** still paying 90 days after the scheduled 14-day trial end, using all randomized customers in each arm as the denominator. **Secondary outcome:** conversion by scheduled trial end. Every arm shares the same follow-up clock. Subscription revenue is prorated over paid days and measured after discounts; recommendation delivery costs are tracked separately. No profit metric is claimed because other operating costs are missing. Milestone 1 summaries are descriptive. Milestone 2 estimates offer-versus-control effects: primary retention comparisons use Bonferroni adjustment across two tests, and exploratory conversion/revenue comparisons use a separate six-test family. No inference claims real-world effectiveness or that an action is worth deploying solely from its sample mean.

## Validate and execute

```bash
python -m pytest -q
python - <<'PY'
import nbformat
from nbclient import NotebookClient
from pathlib import Path
for filename in ['01_synthetic_retention_experiment.ipynb', '02_experiment_analysis.ipynb']:
    path = Path('notebooks') / filename
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute()
    nbformat.write(nb, path)
PY
```

`data/` is wholly ignored, including answer keys and any large regenerated files. Notebook outputs are limited to small previews and aggregates. Do not force-add generated data.

## Repository redesign

The previous CRM notebook referenced a different Parquet path and columns than its generator produced. The old generator simulated raw CRM formatting problems rather than randomized offer outcomes; its validation also assumed a fixed 100,000-customer run. Those files and the old CRM sample were replaced. ClickHouse Docker configuration, ingestion scaffolding, and empty dbt/SQL/config placeholders were removed because they do not serve this milestone. The useful ideas retained are a fictional streaming setting, World Cup dates, seeded generation, validation, and ignoring generated data.

Randomized experiment analysis is now Milestone 2. ML targeting, uplift modeling, and dbt transformations remain later work and are not built here.
