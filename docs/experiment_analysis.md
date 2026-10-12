# Milestone 2 analysis plan

The business question remains: **After the World Cup brings in free-trial customers, which action improves paid retention enough to be worth its cost?**

The primary walkthrough is `notebooks/02_experiment_analysis.ipynb`. It regenerates the unchanged Milestone 1 simulator using 12,000 customers and seed 2026, saves its own exports under ignored `data/milestone2/`, and preserves Milestone 1's files. The simulator's assumptions remain in `docs/assumptions.md`. Statistical helpers in `retention/analysis.py` take observed outcomes only; no predictive model is fitted.

## Population, assignment, and estimands

Eligible customers are uncanceled trial users who remain viewing-active shortly before their scheduled 14-day trial end. Assignment on day 13 is independently randomized with probability 1/3 to control, relevant non-World-Cup content recommendation, or a 20% discount on the first paid month. All groups are followed through scheduled trial end + 90 days.

The estimand is the average effect of **assignment** to each action versus control in the eligible population. The estimator is the offer mean minus the control mean. Every randomized customer stays in their assigned arm; nonconverters and churned customers remain in each denominator. There are no post-treatment exclusions, missing follow-up, noncompliance, or cross-customer interference in this simulator. Random assignment supports the causal interpretation within this invented world; it does not validate the synthetic assumptions or establish real-world effectiveness.

Pre-offer count, mean, standard-deviation, and standardized mean-difference summaries are randomization diagnostics. They do not trigger rerandomization, exclusions, or searches for favorable subgroups. Baseline mean differences are scaled by the pooled within-group standard deviation. No baseline significance testing is used.

## Primary family: two paid-retention comparisons

Outcome: `retained_day90`, using all randomized customers, not retention conditional on conversion. Positive effect means higher paid retention. Effects and intervals are displayed in **percentage points**.

For recommendation versus control and discount versus control:

- Two-sided pooled two-proportion z tests of equal retention probabilities.
- Unpooled Wald confidence intervals for the risk difference: standard error `sqrt(p1(1−p1)/n1 + p0(1−p0)/n0)`.
- Family error level alpha = 0.05; Bonferroni-adjusted p = min(1, 2 × raw p).
- Ordinary 95% pointwise intervals shown for context. Bonferroni-adjusted intervals use a normal critical value at `1 − 0.05/(2×2)`, giving 97.5% marginal confidence and approximately at least 95% simultaneous coverage across the primary family.

Bonferroni remains valid with the shared control; it is conservative. Tests and intervals rely on large-sample approximations. The helper requires at least 10 successes and 10 failures in every arm rather than silently applying the normal method to sparse binary data. The pooled null test and unpooled interval use different standard errors, so their borderline decisions can differ slightly. Neither relative lift nor converter-only retention is substituted for the primary outcome.

## Separate exploratory family: six secondary comparisons

Two action-versus-control comparisons for each of:

1. `converted` by scheduled trial end, displayed in percentage points with the binary method above.
2. `subscription_revenue_90`, displayed in USD per assigned customer.
3. `revenue_less_delivery_cost_90`, displayed in USD per assigned customer.

Conversion is secondary. Revenue tests use two-sided Welch t inference for arithmetic means, with unequal sample variances and Welch–Satterthwaite degrees of freedom. Zero and negative amounts are kept. Large independent groups support approximate mean inference despite the skewed revenue distribution; small samples, heavy tails, missing outcomes, or household dependence would require reconsideration. No revenue log transform changes the target quantity.

Bonferroni adjustment covers **all six secondary hypotheses**, including both binary and revenue tests: adjusted p = min(1, 6 × raw p), with marginal confidence `1 − 0.05/6 = 99.1667%`. These intervals have approximately at least 95% simultaneous coverage within the secondary family. The two revenue measures are related, not independent evidence of replication. This family is separate from the primary family: there is no overall 5% family error guarantee over all eight hypotheses. Secondary results are exploratory and cannot rescue an unsupported primary claim.

## Economic interpretation and uncertainty

Subscription revenue already subtracts discounts from reference revenue. Revenue less delivery cost additionally subtracts $0.10 from every recommendation assignment. Discount-arm delivery costs are zero; discount losses must not be subtracted again. Both metrics use the same 90-day trial-end window and all assigned customers. Acquisition, licensing, baseline hosting, taxes, fixed costs, and other operating costs are omitted. These are revenue measures, not profit, and a positive difference cannot establish profitability.

A higher sample average alone does not establish a winner. Adjusted intervals describe sampling uncertainty; zero-crossing intervals leave both directions plausible. Non-rejection is not proof of zero effect or equivalence. A small p-value is not the probability the null is true or the probability that deployment succeeds. Statistical significance does not establish practical importance. The analysis does not invent a minimum economically worthwhile retention gain, claim superiority between offers, or conduct noninferiority/equivalence tests. Those would require their own thresholds and design.

No optional peeking, hypothesis switching, subgroup searching, ML targeting, uplift modeling, or dbt is added. The milestone ends with average randomized comparisons and their limits.

## Final oracle diagnostic boundary

During generation, the key is written separately and removed from memory without inspection. Observed inference and interpretation finish before the notebook loads `data/milestone2/evaluation_only/oracle_probabilities.csv`.

The diagnostic averages each action's unconditional retention probability over **the same entire generated cohort**, then subtracts the cohort's control probability mean. It compares this planted expected effect with the observed difference and its adjusted interval. Conversion probabilities are checked likewise. Do not average only assigned-arm probabilities from different customer mixes or use conditional retention probabilities as the primary target. The key provides probabilities, not expected revenue amounts or realized counterfactual outcomes.

Observed estimates can differ from the key because of random assignment composition and sampled outcomes. A single interval containing the key does not demonstrate long-run coverage; a miss is not automatically a simulator bug. The diagnostic never changes the inference, selects features, trains a model, or joins oracle values into observed customer features. It is simulation ground truth unavailable in a real experiment.

## Validation and artifacts

Run `python -m pytest -q`, execute the notebook top to bottom, and run `git diff --check`. Tests verify known risk differences, pooled-test calculations, adjusted interval widening and p-values, agreement with SciPy's independent Welch implementation, complete denominators including zero/negative revenue, failure on missing/duplicate/unknown/sparse data, and invariance to oracle columns and row ordering.

Git stores notebook previews, aggregate result tables, and charts. Generated observed rows, oracle rows, run metadata, and CSV result tables stay under ignored `data/`; they are not added to Git.
