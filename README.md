# StreamSphere Customer Decisioning Platform

An end-to-end data, machine learning, and experimentation platform for a
fictional streaming company evaluating whether ML-based offer recommendations
can improve World Cup subscription conversion, revenue, and post-tournament
retention compared with a rule-based strategy.

## Business question

Can an ML-based recommendation strategy generate higher conversion and revenue
than the existing rule-based strategy without increasing customer churn?

## Planned architecture

Salesforce CRM, Stripe billing, Segment behavioral events, playback logs,
Braze marketing events, and content metadata
→ ClickHouse raw layer
→ dbt staging and dimensional models
→ Customer feature tables
→ MLflow model training and registry
→ FastAPI decision service
→ Rule-based versus ML-based A/B experiment
→ Retention and revenue dashboard

## Current status

- Local ClickHouse environment configured with Docker
- Raw Salesforce CRM data generator implemented
- 100,000 synthetic customers with realistic source-system inconsistencies
- Stripe, playback, marketing, and content source generators in development
