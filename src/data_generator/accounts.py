from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker


SEED = 42
NUM_ACCOUNTS = 2_000

fake = Faker()
Faker.seed(SEED)
random.seed(SEED)


def random_signup_date() -> datetime:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    random_days = random.randint(0, (end_date - start_date).days)
    return start_date + timedelta(days=random_days)


def generate_accounts(num_accounts: int = NUM_ACCOUNTS) -> pd.DataFrame:
    records = []

    industries = [
        "Software",
        "Financial Services",
        "Healthcare",
        "Retail",
        "Education",
        "Manufacturing",
        "Media",
        "Professional Services",
    ]

    acquisition_channels = [
        "Organic Search",
        "Paid Search",
        "LinkedIn",
        "Referral",
        "Partner",
        "Direct",
        "Webinar",
    ]

    countries = ["United States", "Canada", "United Kingdom", "Germany", "Australia"]

    company_sizes = ["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"]

    for account_number in range(1, num_accounts + 1):
        signup_date = random_signup_date()

        records.append(
            {
                "account_id": str(uuid.uuid4()),
                "account_number": account_number,
                "company_name": fake.unique.company(),
                "industry": random.choice(industries),
                "company_size": random.choice(company_sizes),
                "country": random.choices(
                    countries,
                    weights=[65, 10, 10, 8, 7],
                    k=1,
                )[0],
                "acquisition_channel": random.choice(acquisition_channels),
                "signup_date": signup_date.date(),
                "created_at": signup_date,
                "is_active": random.random() < 0.88,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    accounts = generate_accounts()

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "accounts.parquet"
    accounts.to_parquet(output_path, index=False)

    print(f"Generated {len(accounts):,} accounts")
    print(f"Saved to: {output_path}")
    print(accounts.head())


if __name__ == "__main__":
    main()