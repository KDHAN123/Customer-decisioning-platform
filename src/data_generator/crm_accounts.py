from __future__ import annotations

import hashlib
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
NUM_CUSTOMERS = 100_000

ACCOUNT_START = datetime(2024, 6, 1)
WORLD_CUP_START = datetime(2026, 6, 11)
WORLD_CUP_END = datetime(2026, 7, 19, 23, 59, 59)

# Salesforce exports can contain repeated records due to sync retries,
# historical migrations, and duplicate imports.
DUPLICATE_RATE = 0.005

UUID_NAMESPACE = uuid.UUID(
    "62ac5151-5cb8-4b08-84ef-23752a4e7610"
)

random.seed(SEED)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def random_datetime_between(
    start: datetime,
    end: datetime,
) -> datetime:
    """Return a random datetime between two boundaries."""
    if start > end:
        raise ValueError("start must occur before end")

    total_seconds = int((end - start).total_seconds())
    offset_seconds = random.randint(0, total_seconds)

    return start + timedelta(seconds=offset_seconds)


def deterministic_uuid(value: str) -> str:
    """Create a reproducible UUID from a stable input value."""
    return str(uuid.uuid5(UUID_NAMESPACE, value))


def hash_email(email: str) -> str:
    """Create a non-reversible email hash for the CRM export."""
    return hashlib.sha256(
        email.lower().encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Account timing
# ---------------------------------------------------------------------------

def random_account_created_at() -> datetime:
    """
    Generate customer account creation dates with higher signup volume
    near and during the World Cup.
    """
    signup_period = random.choices(
        population=[
            "12_to_24_months_before",
            "6_to_12_months_before",
            "1_to_6_months_before",
            "final_30_days_before",
            "during_world_cup",
        ],
        weights=[10, 15, 25, 20, 30],
        k=1,
    )[0]

    if signup_period == "12_to_24_months_before":
        start = ACCOUNT_START
        end = WORLD_CUP_START - timedelta(days=366)

    elif signup_period == "6_to_12_months_before":
        start = WORLD_CUP_START - timedelta(days=365)
        end = WORLD_CUP_START - timedelta(days=181)

    elif signup_period == "1_to_6_months_before":
        start = WORLD_CUP_START - timedelta(days=180)
        end = WORLD_CUP_START - timedelta(days=31)

    elif signup_period == "final_30_days_before":
        start = WORLD_CUP_START - timedelta(days=30)
        end = WORLD_CUP_START - timedelta(seconds=1)

    else:
        start = WORLD_CUP_START
        end = WORLD_CUP_END

    return random_datetime_between(start, end)


# ---------------------------------------------------------------------------
# Source-record origins
# ---------------------------------------------------------------------------

def select_record_origin() -> str:
    """
    Select where the CRM customer record originally entered Salesforce.

    These origins intentionally use different formatting conventions.
    """
    return random.choices(
        population=[
            "web_signup",
            "ios_app",
            "android_app",
            "sports_partner_import",
            "legacy_crm_migration",
        ],
        weights=[36, 20, 18, 16, 10],
        k=1,
    )[0]


# ---------------------------------------------------------------------------
# Canonical business-value selection
# ---------------------------------------------------------------------------

COUNTRIES = [
    "United States",
    "Canada",
    "Mexico",
    "United Kingdom",
    "Germany",
    "France",
    "Brazil",
    "Argentina",
    "Japan",
    "South Korea",
    "Australia",
]

COUNTRY_WEIGHTS = [35, 8, 7, 8, 7, 6, 7, 5, 5, 5, 7]

AGE_GROUPS = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55+",
]

AGE_GROUP_WEIGHTS = [20, 32, 23, 15, 10]

ACQUISITION_CHANNELS = [
    "Organic Search",
    "Paid Search",
    "Social Media",
    "Referral",
    "App Store",
    "Direct",
    "Sports Partnership",
]

ACQUISITION_WEIGHTS = [20, 17, 22, 9, 14, 11, 7]

DEVICE_PREFERENCES = [
    "Smart TV",
    "Mobile",
    "Desktop",
    "Tablet",
    "Streaming Device",
    "Game Console",
]

DEVICE_WEIGHTS = [30, 25, 15, 8, 17, 5]

FAVORITE_TEAMS = [
    "United States",
    "Mexico",
    "Canada",
    "Brazil",
    "Argentina",
    "France",
    "Germany",
    "England",
    "Spain",
    "Portugal",
    "Japan",
    "South Korea",
    "Other",
    "No Preference",
]

FAVORITE_TEAM_WEIGHTS = [
    15,
    8,
    5,
    9,
    8,
    8,
    7,
    7,
    6,
    6,
    5,
    5,
    7,
    4,
]

PREFERRED_LANGUAGES = [
    "English",
    "Spanish",
    "Portuguese",
    "French",
    "German",
    "Japanese",
    "Korean",
]

LANGUAGE_WEIGHTS = [48, 20, 9, 8, 6, 5, 4]


# ---------------------------------------------------------------------------
# Source-specific formatting
# ---------------------------------------------------------------------------

def format_country(
    country: str,
    record_origin: str,
) -> str:
    """Format countries differently based on record origin."""
    variants = {
        "United States": {
            "web_signup": "United States",
            "ios_app": "US",
            "android_app": "USA",
            "sports_partner_import": "U.S.",
            "legacy_crm_migration": "united states",
        },
        "United Kingdom": {
            "web_signup": "United Kingdom",
            "ios_app": "UK",
            "android_app": "GB",
            "sports_partner_import": "Great Britain",
            "legacy_crm_migration": "united kingdom",
        },
        "South Korea": {
            "web_signup": "South Korea",
            "ios_app": "KR",
            "android_app": "Korea, South",
            "sports_partner_import": "Republic of Korea",
            "legacy_crm_migration": "south korea",
        },
    }

    if country in variants:
        return variants[country][record_origin]

    if record_origin == "legacy_crm_migration":
        return country.lower()

    return country


def format_language(
    language: str,
    record_origin: str,
) -> str:
    """Format language values according to the source origin."""
    language_codes = {
        "English": "EN",
        "Spanish": "ES",
        "Portuguese": "PT",
        "French": "FR",
        "German": "DE",
        "Japanese": "JA",
        "Korean": "KO",
    }

    if record_origin in {"ios_app", "android_app"}:
        return language_codes[language]

    if record_origin == "legacy_crm_migration":
        return language.lower()

    return language


def format_acquisition_channel(
    channel: str,
    record_origin: str,
) -> str:
    """Create realistic acquisition-channel inconsistencies."""
    variants = {
        "Organic Search": {
            "web_signup": "Organic Search",
            "ios_app": "organic",
            "android_app": "SEO",
            "sports_partner_import": "Organic",
            "legacy_crm_migration": "organic_search",
        },
        "Paid Search": {
            "web_signup": "Paid Search",
            "ios_app": "paid",
            "android_app": "PPC",
            "sports_partner_import": "Paid Search",
            "legacy_crm_migration": "paid_search",
        },
        "Social Media": {
            "web_signup": "Social Media",
            "ios_app": "social",
            "android_app": "SOCIAL",
            "sports_partner_import": "Social Partner",
            "legacy_crm_migration": "social_media",
        },
        "App Store": {
            "web_signup": "App Store",
            "ios_app": "Apple App Store",
            "android_app": "Google Play",
            "sports_partner_import": "Mobile App",
            "legacy_crm_migration": "app_store",
        },
        "Sports Partnership": {
            "web_signup": "Sports Partnership",
            "ios_app": "sports_partner",
            "android_app": "SPORTS_PARTNER",
            "sports_partner_import": "Partner",
            "legacy_crm_migration": "sports partnership",
        },
    }

    if channel in variants:
        return variants[channel][record_origin]

    if record_origin == "legacy_crm_migration":
        return channel.lower().replace(" ", "_")

    return channel


def format_device(
    device: str,
    record_origin: str,
) -> str:
    """Format device preferences according to record origin."""
    if record_origin == "ios_app":
        if device == "Mobile":
            return "iPhone"

        if device == "Tablet":
            return "iPad"

    if record_origin == "android_app":
        if device == "Mobile":
            return "Android Phone"

        if device == "Tablet":
            return "Android Tablet"

    if record_origin == "legacy_crm_migration":
        return device.upper().replace(" ", "_")

    return device


def format_marketing_consent(
    consent: bool,
    record_origin: str,
) -> str:
    """Represent consent using source-specific string formats."""
    formats = {
        "web_signup": {
            True: "true",
            False: "false",
        },
        "ios_app": {
            True: "1",
            False: "0",
        },
        "android_app": {
            True: "TRUE",
            False: "FALSE",
        },
        "sports_partner_import": {
            True: "Yes",
            False: "No",
        },
        "legacy_crm_migration": {
            True: "Y",
            False: "N",
        },
    }

    return formats[record_origin][consent]


def format_timestamp(
    value: datetime,
    record_origin: str,
) -> str:
    """Represent timestamps using source-specific formats."""
    if record_origin == "web_signup":
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")

    if record_origin == "ios_app":
        return value.strftime("%Y-%m-%d %H:%M:%S")

    if record_origin == "android_app":
        return value.strftime("%m/%d/%Y %I:%M %p")

    if record_origin == "sports_partner_import":
        return value.strftime("%Y/%m/%d %H:%M")

    return value.strftime("%d-%b-%Y %H:%M:%S")


def format_household_size(
    household_size: int,
    record_origin: str,
) -> str:
    """Represent household size as raw source text."""
    if record_origin == "legacy_crm_migration":
        return str(household_size)

    if record_origin == "sports_partner_import":
        return f"{household_size}.0"

    return str(household_size)


# ---------------------------------------------------------------------------
# Realistic raw-data problems
# ---------------------------------------------------------------------------

def maybe_add_whitespace(value: str) -> str:
    """Add accidental whitespace to a small percentage of values."""
    if value and random.random() < 0.025:
        return f"  {value} "

    return value


def maybe_missing(
    value: str,
    probability: float,
) -> str | None:
    """Replace an optional value with a blank or null."""
    if random.random() >= probability:
        return value

    return random.choice(["", None])


def maybe_invalid_household_size(value: str) -> str:
    """Create a small number of invalid household-size values."""
    if random.random() < 0.002:
        return random.choice(
            [
                "0",
                "-1",
                "99",
                "unknown",
                "N/A",
            ]
        )

    return value


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_crm_accounts(
    num_customers: int = NUM_CUSTOMERS,
) -> pd.DataFrame:
    """
    Generate a raw Salesforce-style CRM account export.

    The records are intentionally inconsistent because they entered the CRM
    through different operational channels.
    """
    records: list[dict[str, object]] = []

    extract_timestamp = datetime(
        2026,
        7,
        20,
        6,
        0,
        0,
    )

    for customer_number in range(1, num_customers + 1):
        external_user_id = f"usr_{customer_number:08d}"

        crm_account_id = deterministic_uuid(
            f"salesforce-account-{customer_number}"
        )

        email = (
            f"customer{customer_number:08d}"
            "@streamsphere.example"
        )

        record_origin = select_record_origin()
        account_created_at = random_account_created_at()

        country = random.choices(
            COUNTRIES,
            weights=COUNTRY_WEIGHTS,
            k=1,
        )[0]

        age_group = random.choices(
            AGE_GROUPS,
            weights=AGE_GROUP_WEIGHTS,
            k=1,
        )[0]

        acquisition_channel = random.choices(
            ACQUISITION_CHANNELS,
            weights=ACQUISITION_WEIGHTS,
            k=1,
        )[0]

        device_preference = random.choices(
            DEVICE_PREFERENCES,
            weights=DEVICE_WEIGHTS,
            k=1,
        )[0]

        favorite_team = random.choices(
            FAVORITE_TEAMS,
            weights=FAVORITE_TEAM_WEIGHTS,
            k=1,
        )[0]

        preferred_language = random.choices(
            PREFERRED_LANGUAGES,
            weights=LANGUAGE_WEIGHTS,
            k=1,
        )[0]

        household_size = random.choices(
            [1, 2, 3, 4, 5, 6],
            weights=[20, 30, 20, 17, 8, 5],
            k=1,
        )[0]

        marketing_consent = random.random() < 0.72

        raw_country = format_country(
            country,
            record_origin,
        )

        raw_language = format_language(
            preferred_language,
            record_origin,
        )

        raw_channel = format_acquisition_channel(
            acquisition_channel,
            record_origin,
        )

        raw_device = format_device(
            device_preference,
            record_origin,
        )

        raw_household_size = format_household_size(
            household_size,
            record_origin,
        )

        record = {
            "crm_account_id": crm_account_id,
            "external_user_id": external_user_id,
            "email_hash": hash_email(email),
            "record_origin": record_origin,
            "country": maybe_add_whitespace(raw_country),
            "age_group": maybe_missing(
                maybe_add_whitespace(age_group),
                probability=0.015,
            ),
            "acquisition_channel": maybe_add_whitespace(
                raw_channel
            ),
            "device_preference": maybe_missing(
                maybe_add_whitespace(raw_device),
                probability=0.025,
            ),
            "favorite_team": maybe_missing(
                maybe_add_whitespace(favorite_team),
                probability=0.035,
            ),
            "preferred_language": maybe_missing(
                maybe_add_whitespace(raw_language),
                probability=0.02,
            ),
            "household_size": maybe_invalid_household_size(
                raw_household_size
            ),
            "marketing_consent": format_marketing_consent(
                marketing_consent,
                record_origin,
            ),
            "account_created_at": format_timestamp(
                account_created_at,
                record_origin,
            ),
            "last_modified_at": format_timestamp(
                account_created_at
                + timedelta(
                    days=random.randint(0, 90),
                    seconds=random.randint(0, 86_399),
                ),
                record_origin,
            ),
            "salesforce_extract_at": extract_timestamp.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }

        records.append(record)

    crm_accounts = pd.DataFrame(records)

    # Simulate duplicate records created by source synchronization retries.
    duplicate_count = int(
        num_customers * DUPLICATE_RATE
    )

    duplicate_records = crm_accounts.sample(
        n=duplicate_count,
        random_state=SEED,
    ).copy()

    crm_accounts = pd.concat(
        [crm_accounts, duplicate_records],
        ignore_index=True,
    )

    # Shuffle the extract so duplicates are not located together.
    crm_accounts = crm_accounts.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    return crm_accounts


# ---------------------------------------------------------------------------
# Raw-data validation
# ---------------------------------------------------------------------------

def validate_raw_crm_accounts(
    crm_accounts: pd.DataFrame,
) -> None:
    """
    Validate the shape of the raw extract.

    Raw-data validation intentionally does not reject duplicates, missing
    optional values, inconsistent labels, or malformed optional fields.
    Those problems will be handled later by dbt.
    """
    expected_rows = NUM_CUSTOMERS + int(
        NUM_CUSTOMERS * DUPLICATE_RATE
    )

    if len(crm_accounts) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} raw rows, "
            f"found {len(crm_accounts):,}"
        )

    required_columns = [
        "crm_account_id",
        "external_user_id",
        "email_hash",
        "record_origin",
        "account_created_at",
        "salesforce_extract_at",
    ]

    missing_required_columns = [
        column
        for column in required_columns
        if column not in crm_accounts.columns
    ]

    if missing_required_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_required_columns}"
        )

    for column in required_columns:
        if crm_accounts[column].isna().any():
            raise ValueError(
                f"Required column contains nulls: {column}"
            )

        if (
            crm_accounts[column]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):
            raise ValueError(
                f"Required column contains blanks: {column}"
            )

    unique_customer_count = (
        crm_accounts["external_user_id"].nunique()
    )

    if unique_customer_count != NUM_CUSTOMERS:
        raise ValueError(
            f"Expected {NUM_CUSTOMERS:,} unique customers, "
            f"found {unique_customer_count:,}"
        )

    duplicate_count = crm_accounts.duplicated(
        subset=["crm_account_id"],
        keep=False,
    ).sum()

    if duplicate_count == 0:
        raise ValueError(
            "Expected duplicate CRM records, but none were generated"
        )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def main() -> None:
    crm_accounts = generate_crm_accounts()
    validate_raw_crm_accounts(crm_accounts)

    output_dir = Path(
        "data/raw/salesforce_crm"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir / "crm_accounts.parquet"
    )

    crm_accounts.to_parquet(
        output_path,
        index=False,
    )

    sample_dir = Path("data/samples")
    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_path = (
        sample_dir / "crm_accounts_sample.csv"
    )

    crm_accounts.head(200).to_csv(
        sample_path,
        index=False,
    )

    print(
        f"Generated {len(crm_accounts):,} raw CRM rows"
    )
    print(
        f"Unique customers: "
        f"{crm_accounts['external_user_id'].nunique():,}"
    )
    print(
        f"Duplicate rows added: "
        f"{len(crm_accounts) - NUM_CUSTOMERS:,}"
    )
    print(f"Raw file saved to: {output_path}")
    print(f"GitHub sample saved to: {sample_path}")
    print()

    print("Record origins:")
    print(
        crm_accounts["record_origin"]
        .value_counts()
    )
    print()

    print("Raw country examples:")
    print(
        crm_accounts["country"]
        .value_counts(dropna=False)
        .head(20)
    )
    print()

    print("Missing values:")
    print(
        crm_accounts.isna().sum()
    )
    print()

    print("Sample records:")
    print(
        crm_accounts.head()
    )


if __name__ == "__main__":
    main()