# Synthetic experiment assumptions — Milestone 1

**Business question: After the World Cup brings in free-trial customers, which action improves paid retention enough to be worth its cost?**

This document specifies the invented world implemented in `retention/simulation.py` and explained in the notebook. None of its response patterns is a discovery about real customers. There is no ML targeting, uplift model, formal A/B analysis, or dbt implementation in this milestone.

## Eligible population and timeline

One row represents one randomized customer. Signups fall May 28–July 5, 2026, around the World Cup (June 11–July 19). World-Cup-related content includes pre-tournament coverage and previews, not just live matches. This is a fictional licensed catalog, not a reconstruction of viewing schedules.

A 14-day trial has a fixed scheduled end. At the start of trial day 13, eligibility requires an uncanceled trial and at least one viewing-active day in the preceding three completed days. Offers are delivered on day 13, one day before scheduled trial end. Features summarize the 13 completed days in `[signup_date, assignment_date)`; `feature_window_end_date` is the exclusive cutoff. There is no offer-period behavior in the feature allowlist.

The generator creates **12,000 eligible customers by default**, conditional on this definition. It does not simulate all signups and then screen them. `active_at_offer`, `canceled_before_offer`, and `eligible_at_offer` explicitly record the conditioning; these constant audit flags are not ML features. Early cancellations, inactive trial users, prior subscribers, and repeat accounts are outside this cohort. Conclusions from a future experiment would apply to eligible customers, not all signups.

An independent random stream assigns each eligible customer to exactly one of three arms with probability **1/3**. There is no behavior-based assignment, exact-count balancing, noncompliance, cross-customer interference, or later rerandomization.

| Arm | Policy | Incremental economic assumption |
|---|---|---|
| `control` | No special offer; standard $15/month subscription | No offer cost |
| `content_recommendation` | Recommend relevant non-World-Cup movies, series, and other continuing content; same plan and price | $0.10 delivery cost for every assigned customer, including nonconverters |
| `discount_20pct` | 20% discount on first paid month; standard price thereafter | Forgone subscription revenue only; no additional delivery cost |

## Outcomes and common clock

**Primary outcome:** `retained_day90` is true only if the customer converts and is still paying 90 days after `scheduled_trial_end_date`. Use **all randomized customers within each arm** as the denominator. Nonconverters count as not retained.

**Secondary outcome:** `converted` means converted by scheduled trial end. The simplified generator places all successful paid starts at scheduled trial end (`conversion_day = 0`, measured relative to that end); others never convert later. Conversion is not the primary success metric.

All arms have the same 90-day paid observation window, `[scheduled_trial_end_date, followup_end_date)`. There are no extensions, delayed paid starts, censored customers, refunds, billing failures, or reactivations. Conditional daily churn is constant rather than spiking at tournament end. A geometric cancellation time `k` counts the paid daily intervals before cancellation; survival beyond 90 intervals means retained. Cancellation exactly at the day-90 boundary is not retained, but can still accrue 90 paid days.

## Pre-offer behavior and planted response patterns

Three latent preferences are drawn independently: engagement ~ Beta(2,2), World Cup interest ~ Beta(3,2), and price sensitivity ~ Beta(2,2). They drive the simulator but are never exported as model features. Pricing-page visits are only a noisy observable proxy for price sensitivity.

Let `e` = engagement, `w` = World Cup interest, and `s` = price sensitivity.

| Observable pre-offer feature | Generation assumption |
|---|---|
| `active_days_last3_pre` | 1 + Binomial(2, 0.15 + 0.70e), ensuring eligibility |
| `active_days_pre` | Binomial(10, 0.15 + 0.70e) + recent active days; at most 13 |
| `watch_hours_pre` | Active days × Gamma(shape 2, scale 0.65), rounded to 2 decimals |
| `world_cup_share_pre` | `w`, rounded to 3 decimals; World-Cup-related share of viewing |
| `non_world_cup_hours_pre` | Watch hours × (1 − World Cup share), rounded to 2 decimals |
| `content_breadth_pre` | 1 + Binomial(4, 1 − w) if non-World-Cup hours > 0, otherwise 0; number of distinct non-World-Cup categories watched |
| `pricing_page_visits_pre` | Poisson(0.3 + 2.5s) |
| `discovery_sessions_pre` | Binomial(active days, 1 − w); sessions exploring beyond World Cup content |

The activity distribution is a direct approximation for the eligible population, rather than exact rejection sampling from a population of all trials. Categories and exploration sessions are coarse summaries, not a full playback-event simulation.

Baseline conversion log-odds = `−1.5 + 0.15 × active days + 0.025 × watch hours − 0.9s`.

Baseline conditional day-90 retention log-odds = `−0.2 + 0.12 × active days − 1.3w`.

| Arm | Conversion log-odds addition | Conditional retention log-odds addition |
|---|---|---|
| No special offer | 0 | 0 |
| Content recommendation | `1.2(w − 0.55)` | `2.2(w − 0.55)` |
| Discount | `1.5(s − 0.60)` | `0.7(s − 0.60)` |

Log-odds become probabilities via `1 / (1 + exp(−score))`. Recommendations help World-Cup-heavy users (`w > 0.55`) discover continuing value; they can irritate or distract users who already explore other content (`w < 0.55`). Discounts help sufficiently price-sensitive users (`s > 0.60`), but can create skepticism or an expectation of promotional pricing for others. At either threshold the corresponding planted effect is zero. Both beneficial and harmful effects are allowed; no average winner is guaranteed. These mechanisms, coefficients, and thresholds are **planted assumptions**, not learned findings.

Conversion is sampled as a Bernoulli draw under the assigned arm. Conditional retention probability is translated to a daily hazard `1 − p_conditional ** (1/90)`; a geometric draw produces cancellation time. Only the assigned arm's realized outcomes are observed.

## Revenue and cost accounting

Amounts are fictional USD. Months have 30 days and billing is prorated ($15/30 = $0.50/day), rather than upfront nonrefundable monthly charges.

- `reference_revenue_90` = paid days × $0.50, a reference amount before discounts.
- `discount_cost_90` = 20% × $0.50 × min(30, paid days) for discount-arm converters, zero otherwise. Maximum $3 per converted customer.
- `subscription_revenue_90` = reference revenue − discount cost. This is subscription **revenue after discounts**.
- `delivery_cost_90` = $0.10 for every recommendation assignment, zero otherwise.
- `offer_cost_90` = discount cost + delivery cost, exposing the economic burden of each action.
- `revenue_less_delivery_cost_90` = subscription revenue − delivery cost = reference revenue − offer cost. This supplementary ledger can be negative for recommendation nonconverters. It is **not profit**.

Do not subtract total offer cost from subscription revenue: that would subtract the discount twice. Acquisition, licensing, baseline hosting, taxes, fixed costs, and other operating expenses are missing, so there is no profit metric. Retention, revenue, and offer costs establish the tradeoff for later analysis; this milestone does not decide whether an action is worth deploying.

## Oracle separation and later evaluation

`data/generated/customers.csv` stores observed data. `PRE_OFFER_FEATURES` explicitly allows only the behavioral columns above for later model inputs. IDs, audit flags, assignment, dates, paid days, costs, and outcomes serve joins, audits, or labels; they are excluded from that allowlist.

`data/evaluation_only/oracle_probabilities.csv` is a separate three-row-per-customer answer key: conversion probability, day-90 retention probability conditional on conversion, and unconditional retention probability (their product) for each possible action. It is **simulation ground truth for later method checks**, not a feature available to a real prediction model and not realized counterfactual outcomes. Latent preferences are not exported into the observed or feature datasets.

The notebook briefly joins assigned-arm probabilities into an evaluation-only diagnostic table to check sampling calibration, then deletes that table. It never joins the oracle into customer feature data. Generated datasets and run metadata remain ignored by Git. Fixed seeds reproduce results within the same package versions; metadata records versions and the scenario identifier.

## Analysis of this unchanged scenario

Milestone 2 (`notebooks/02_experiment_analysis.ipynb`) analyzes these same generated outcomes without changing the simulator. Its tests, confidence intervals, and multiplicity plan are documented in `docs/experiment_analysis.md`. This document specifies Milestone 1 generation assumptions; Milestone 2 adds observed-data inference, followed by a separate final oracle method check.
