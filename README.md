# Customer Decisioning & Experimentation Platform

An end-to-end analytics and machine learning platform that evaluates whether an ML-based customer offer strategy can outperform an existing rule-based strategy in SaaS trial conversion and revenue.

## Planned architecture

Synthetic customer and behavioral data  
→ ClickHouse raw warehouse  
→ dbt SQL transformations  
→ Star-schema analytics layer  
→ Customer feature tables  
→ ML model training and MLflow  
→ FastAPI decision service  
→ A/B experiment simulation  
→ Statistical analysis  
→ Superset dashboard

## Current status

Project environment and warehouse foundation in progress.
