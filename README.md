# World Cup Streaming Decisioning Platform

An end-to-end customer decisioning and experimentation platform that evaluates
whether an ML-based subscription-offer strategy can outperform a rule-based
strategy during a major streaming event such as the FIFA World Cup.

The platform generates synthetic users, match-viewing behavior, subscription
activity, offer interactions, purchases, cancellations, and experiment outcomes.


# Planned Architecture

Synthetic users, matches, and streaming events
→ ClickHouse raw event storage
→ dbt transformations
→ Customer feature table
→ ML model training and MLflow
→ FastAPI recommendation service
→ Rule-based vs ML-based A/B experiment
→ Subscription, revenue, cancellation, and retention analysis
→ Experiment dashboard
