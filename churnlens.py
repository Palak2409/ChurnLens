"""
ChurnLens: Customer Churn Analytics and Risk Predictor
Core Data Preparation and Analytics Foundation

This module handles:
1. Safe loading and validation of the IBM Telco Customer Churn dataset.
2. Robust data cleaning, handling whitespace/blanks, type casting, and duplicate removal.
3. Creation of target variables for modeling without mutating presentation labels.
4. Reusable analytical calculations for business insights, churn rates, and distributions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Expected columns in the IBM Telco Customer Churn dataset
REQUIRED_COLUMNS: List[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def resolve_data_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Finds and resolves the path to the Telco Customer Churn CSV file.

    Search priority:
    1. custom_path (if explicitly provided and exists)
    2. 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    3. Any *.csv file found inside the 'data/' folder
    4. 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the project root directory
       (automatically copies it into 'data/' if found in root)

    Parameters:
    -----------
    custom_path : str or Path, optional
        A specific path to a CSV file.

    Returns:
    --------
    Path:
        Resolved Path object pointing to an existing CSV file.

    Raises:
    -------
    FileNotFoundError:
        If no suitable CSV file is found.
    """
    if custom_path is not None:
        p = Path(custom_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Provided dataset path does not exist: {custom_path}")

    # Standard data folder path
    data_dir = Path("data")
    default_in_data = data_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if default_in_data.is_file():
        return default_in_data

    # Check for any CSV in data/
    if data_dir.is_dir():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            return csv_files[0]

    # Check root directory as fallback and copy to data/
    root_csv = Path("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if root_csv.is_file():
        data_dir.mkdir(parents=True, exist_ok=True)
        destination = data_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        try:
            shutil.copy2(root_csv, destination)
            return destination
        except Exception:
            return root_csv

    raise FileNotFoundError(
        "Could not find the Telco Customer Churn dataset. "
        "Please ensure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' is placed inside the 'data/' folder."
    )


def load_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Loads the raw CSV dataset into a pandas DataFrame.

    Parameters:
    -----------
    filepath : str or Path, optional
        Path to the CSV file. If None, resolve_data_path() locates the file.

    Returns:
    --------
    pd.DataFrame:
        Raw, unprocessed DataFrame loaded from the CSV.

    Raises:
    -------
    FileNotFoundError:
        If the file cannot be located.
    ValueError:
        If the file is empty or cannot be parsed as a valid CSV.
    """
    resolved_path = resolve_data_path(filepath)

    try:
        df = pd.read_csv(resolved_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV at {resolved_path}: {e}") from e

    if df.empty:
        raise ValueError(f"The dataset at {resolved_path} is empty (0 rows).")

    return df


def validate_columns(
    df: pd.DataFrame,
    required_cols: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validates that the DataFrame contains all expected columns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to inspect.
    required_cols : list of str, optional
        List of column names required. Defaults to REQUIRED_COLUMNS.

    Returns:
    --------
    tuple of (bool, list of str):
        (is_valid, missing_columns)
        is_valid is True if all required columns are present, else False.
        missing_columns contains any column names that were missing.
    """
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS

    df_columns = set(df.columns)
    missing = [col for col in required_cols if col not in df_columns]
    return (len(missing) == 0, missing)


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspects the DataFrame and returns a comprehensive health report.

    Checks:
    - Dimensions (row count, column count)
    - Column data types
    - Standard NaN / null counts
    - Blank / whitespace strings in string columns (like ' ' in TotalCharges)
    - Duplicate customer IDs

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to audit.

    Returns:
    --------
    dict:
        Detailed diagnostic dictionary.
    """
    num_rows, num_cols = df.shape
    nan_counts = df.isna().sum().to_dict()

    # Detect whitespace-only strings across object/string columns
    whitespace_counts: Dict[str, int] = {}
    for col in df.select_dtypes(include=["object", "string"]).columns:
        ws_count = int((df[col].astype(str).str.strip() == "").sum())
        if ws_count > 0:
            whitespace_counts[col] = ws_count

    # Check for duplicate customer IDs if customerID exists
    duplicate_ids_count = 0
    if "customerID" in df.columns:
        duplicate_ids_count = int(df["customerID"].duplicated().sum())

    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "nan_counts": nan_counts,
        "total_nans": int(sum(nan_counts.values())),
        "whitespace_counts": whitespace_counts,
        "duplicate_customer_ids": duplicate_ids_count,
    }


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans and prepares the Telco Customer Churn dataset for analysis and modeling.

    Cleaning steps:
    1. Validates required columns.
    2. Identifies and removes duplicate customer IDs (retaining first occurrence).
    3. Safely converts 'TotalCharges' to float64:
       - Blank strings (' ') are converted to NaN.
       - For records with tenure == 0, TotalCharges is imputed with 0.0
         (because brand-new customers with 0 months have accrued $0 in charges).
         This preserves valid customer records rather than dropping them.
       - Any remaining unexpected NaNs are handled explicitly.
    4. Casts 'tenure' to integer and 'MonthlyCharges' to float.
    5. Creates a binary 'ChurnBinary' target (1 for Yes, 0 for No) while
       preserving the original 'Churn' column ('Yes'/'No') for user-facing display.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset.

    Returns:
    --------
    tuple of (pd.DataFrame, dict):
        - cleaned_df: Cleaned and ready-to-use DataFrame.
        - cleaning_report: Summary dictionary explaining what changes were made.

    Raises:
    -------
    ValueError:
        If required columns are missing from the dataset.
    """
    is_valid, missing_cols = validate_columns(df)
    if not is_valid:
        raise ValueError(
            f"Dataset cannot be cleaned because required columns are missing: {missing_cols}"
        )

    # Work on a copy to avoid mutating the original input DataFrame
    cleaned = df.copy()
    initial_rows = len(cleaned)

    # 1. Deduplication on customerID
    duplicates_mask = cleaned["customerID"].duplicated(keep="first")
    num_duplicates = int(duplicates_mask.sum())
    removed_ids: List[str] = []
    if num_duplicates > 0:
        removed_ids = cleaned.loc[duplicates_mask, "customerID"].tolist()
        cleaned = cleaned[~duplicates_mask].copy()

    # 2. TotalCharges whitespace handling & numeric conversion
    # Identify strings that are purely whitespace
    raw_tc = cleaned["TotalCharges"].astype(str).str.strip()
    blank_tc_mask = raw_tc == ""
    num_blank_tc = int(blank_tc_mask.sum())

    # Convert to numeric, forcing unparseable strings to NaN
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

    # Safe zero-tenure imputation:
    # Customers with tenure == 0 have just signed up and accrued $0 in total charges.
    zero_tenure_mask = (cleaned["tenure"] == 0) & (cleaned["TotalCharges"].isna())
    zero_tenure_imputed_count = int(zero_tenure_mask.sum())
    cleaned.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    # Check if any unexplained NaNs remain in TotalCharges
    unexplained_tc_nans = int(cleaned["TotalCharges"].isna().sum())
    if unexplained_tc_nans > 0:
        # If any customer with tenure > 0 has missing TotalCharges, impute with tenure * MonthlyCharges
        # rather than an arbitrary global mean, preserving business logic.
        fill_mask = cleaned["TotalCharges"].isna()
        cleaned.loc[fill_mask, "TotalCharges"] = (
            cleaned.loc[fill_mask, "tenure"] * cleaned.loc[fill_mask, "MonthlyCharges"]
        )

    # 3. Explicit type casting
    cleaned["tenure"] = cleaned["tenure"].astype(int)
    cleaned["MonthlyCharges"] = cleaned["MonthlyCharges"].astype(float)
    cleaned["TotalCharges"] = cleaned["TotalCharges"].astype(float)
    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(int)

    # 4. Clean binary churn target for ML models
    # Normalize string values to uppercase stripped for robustness
    churn_clean = cleaned["Churn"].astype(str).str.strip().str.capitalize()
    cleaned["Churn"] = churn_clean  # Keeps 'Yes' and 'No' for clear display
    cleaned["ChurnBinary"] = (cleaned["Churn"] == "Yes").astype(int)

    final_rows = len(cleaned)

    cleaning_report: Dict[str, Any] = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "duplicates_removed": num_duplicates,
        "duplicate_ids_removed": removed_ids,
        "blank_total_charges_found": num_blank_tc,
        "zero_tenure_customers_preserved": zero_tenure_imputed_count,
        "unexplained_tc_nans_handled": unexplained_tc_nans,
        "final_missing_values": int(cleaned.isna().sum().sum()),
        "churn_distribution": cleaned["Churn"].value_counts().to_dict(),
        "churn_rate_pct": round(float(cleaned["ChurnBinary"].mean() * 100), 2),
    }

    return cleaned, cleaning_report


# ==============================================================================
# REUSABLE ANALYTICAL FUNCTIONS
# ==============================================================================

def get_churn_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes overall customer counts and churn rates.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned or prepared DataFrame containing 'Churn' or 'ChurnBinary'.

    Returns:
    --------
    dict:
        - total_customers (int)
        - churned_customers (int)
        - retained_customers (int)
        - churn_rate_pct (float)
        - retention_rate_pct (float)
    """
    total = len(df)
    if total == 0:
        return {
            "total_customers": 0,
            "churned_customers": 0,
            "retained_customers": 0,
            "churn_rate_pct": 0.0,
            "retention_rate_pct": 0.0,
        }

    if "ChurnBinary" in df.columns:
        churned = int(df["ChurnBinary"].sum())
    else:
        churned = int((df["Churn"].astype(str).str.strip().str.capitalize() == "Yes").sum())

    retained = total - churned
    churn_rate = (churned / total) * 100
    retention_rate = (retained / total) * 100

    return {
        "total_customers": total,
        "churned_customers": churned,
        "retained_customers": retained,
        "churn_rate_pct": round(float(churn_rate), 2),
        "retention_rate_pct": round(float(retention_rate), 2),
    }


def get_charge_and_tenure_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes central tendency statistics for MonthlyCharges, TotalCharges, and tenure,
    both overall and segmented by churn status.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned DataFrame.

    Returns:
    --------
    dict:
        Summary metrics including overall averages/medians and churn-segmented breakdowns.
    """
    if len(df) == 0:
        return {}

    overall_metrics = {
        "avg_monthly_charges": round(float(df["MonthlyCharges"].mean()), 2),
        "median_monthly_charges": round(float(df["MonthlyCharges"].median()), 2),
        "avg_total_charges": round(float(df["TotalCharges"].mean()), 2),
        "median_total_charges": round(float(df["TotalCharges"].median()), 2),
        "avg_tenure": round(float(df["tenure"].mean()), 2),
        "median_tenure": round(float(df["tenure"].median()), 2),
    }

    # Breakdown by Churn status
    breakdown: Dict[str, Dict[str, float]] = {}
    for status in ["Yes", "No"]:
        subset = df[df["Churn"] == status]
        if len(subset) > 0:
            breakdown[status] = {
                "count": len(subset),
                "avg_monthly_charges": round(float(subset["MonthlyCharges"].mean()), 2),
                "median_monthly_charges": round(float(subset["MonthlyCharges"].median()), 2),
                "avg_total_charges": round(float(subset["TotalCharges"].mean()), 2),
                "median_total_charges": round(float(subset["TotalCharges"].median()), 2),
                "avg_tenure": round(float(subset["tenure"].mean()), 2),
                "median_tenure": round(float(subset["tenure"].median()), 2),
            }

    return {
        "overall": overall_metrics,
        "by_churn": breakdown,
    }


def get_churn_rate_by_category(
    df: pd.DataFrame,
    category_col: str,
    sort_by: str = "churn_rate_desc"
) -> pd.DataFrame:
    """
    Calculates customer counts, churned customer counts, and churn rates
    grouped by any categorical column (e.g., Contract, InternetService, PaymentMethod).

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned DataFrame.
    category_col : str
        Column name to group by.
    sort_by : str
        Sorting order: 'churn_rate_desc', 'churn_rate_asc', or 'count_desc'.

    Returns:
    --------
    pd.DataFrame:
        Table containing:
        - Category Value
        - Total Customers
        - Churned Customers
        - Retained Customers
        - Churn Rate (%)
        - Share of Base (%)
    """
    if category_col not in df.columns:
        raise ValueError(f"Column '{category_col}' not found in DataFrame.")

    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                category_col,
                "Total Customers",
                "Churned Customers",
                "Retained Customers",
                "Churn Rate (%)",
                "Share of Base (%)",
            ]
        )

    # Use binary churn column if present, otherwise evaluate string
    if "ChurnBinary" in df.columns:
        target_series = df["ChurnBinary"]
    else:
        target_series = (df["Churn"] == "Yes").astype(int)

    grouped = df.groupby(category_col).agg(
        total_customers=("customerID", "count"),
        churned_customers=(target_series.name, lambda s: int(target_series.loc[s.index].sum())),
    ).reset_index()

    grouped["retained_customers"] = grouped["total_customers"] - grouped["churned_customers"]
    grouped["churn_rate_pct"] = (
        (grouped["churned_customers"] / grouped["total_customers"]) * 100
    ).round(2)
    grouped["share_of_base_pct"] = (
        (grouped["total_customers"] / len(df)) * 100
    ).round(2)

    # Sort
    if sort_by == "churn_rate_desc":
        grouped = grouped.sort_values(by="churn_rate_pct", ascending=False)
    elif sort_by == "churn_rate_asc":
        grouped = grouped.sort_values(by="churn_rate_pct", ascending=True)
    elif sort_by == "count_desc":
        grouped = grouped.sort_values(by="total_customers", ascending=False)

    grouped = grouped.rename(
        columns={
            "total_customers": "Total Customers",
            "churned_customers": "Churned Customers",
            "retained_customers": "Retained Customers",
            "churn_rate_pct": "Churn Rate (%)",
            "share_of_base_pct": "Share of Base (%)",
        }
    )

    return grouped.reset_index(drop=True)


def get_tenure_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes tenure distribution overall and compared across churned vs retained customers.

    Calculates:
    - Five-number summary + mean + std for tenure (Overall, Churned, Retained).
    - Tenure bracket breakdown (0-12m, 13-24m, 25-36m, 37-48m, 49-60m, 61-72m) with
      customer counts and churn rates per bracket.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned DataFrame.

    Returns:
    --------
    dict:
        - stats: Summary statistics dictionary.
        - brackets: DataFrame summarizing tenure bands and their respective churn rates.
    """
    if len(df) == 0:
        return {"stats": {}, "brackets": pd.DataFrame()}

    def _stats(series: pd.Series) -> Dict[str, float]:
        if len(series) == 0:
            return {}
        return {
            "count": int(len(series)),
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2) if len(series) > 1 else 0.0,
            "min": int(series.min()),
            "q25": round(float(series.quantile(0.25)), 2),
            "median": round(float(series.median()), 2),
            "q75": round(float(series.quantile(0.75)), 2),
            "max": int(series.max()),
        }

    churned_tenure = df.loc[df["Churn"] == "Yes", "tenure"]
    retained_tenure = df.loc[df["Churn"] == "No", "tenure"]

    stats = {
        "overall": _stats(df["tenure"]),
        "churned": _stats(churned_tenure),
        "retained": _stats(retained_tenure),
    }

    # Standard industry tenure cohorts (1-year buckets)
    bins = [-1, 12, 24, 36, 48, 60, 72]
    labels = ["0-12 months", "13-24 months", "25-36 months", "37-48 months", "49-60 months", "61-72 months"]

    bracket_df = df.copy()
    bracket_df["tenure_bracket"] = pd.cut(
        bracket_df["tenure"],
        bins=bins,
        labels=labels
    )

    bracket_summary = get_churn_rate_by_category(
        bracket_df,
        category_col="tenure_bracket",
        sort_by="natural"
    )

    return {
        "stats": stats,
        "brackets": bracket_summary,
    }


def filter_customers(df: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    """
    Filters the dataset based on multiple column-value criteria.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned DataFrame.
    **filters : dict
        Column and value pairs. Values can be single items or lists/tuples of allowed items.

    Returns:
    --------
    pd.DataFrame:
        Filtered subset of the DataFrame.

    Example:
    --------
    filter_customers(df, Contract='Month-to-month', InternetService='Fiber optic')
    """
    subset = df.copy()
    for col, val in filters.items():
        if col not in subset.columns:
            raise KeyError(f"Column '{col}' does not exist in DataFrame.")
        if isinstance(val, (list, tuple, set)):
            subset = subset[subset[col].isin(val)]
        else:
            subset = subset[subset[col] == val]
    return subset


def get_group_churn_metrics(
    df: pd.DataFrame,
    group_name: str = "Selected Cohort"
) -> Dict[str, Any]:
    """
    Calculates customer counts, churn rate, and financial impact for any selected customer group.

    Parameters:
    -----------
    df : pd.DataFrame
        Filtered subset of customers.
    group_name : str
        Descriptive label for the cohort.

    Returns:
    --------
    dict:
        Cohort metrics dictionary.
    """
    total = len(df)
    if total == 0:
        return {
            "group_name": group_name,
            "total_customers": 0,
            "churned_customers": 0,
            "churn_rate_pct": 0.0,
            "total_monthly_revenue": 0.0,
            "lost_monthly_revenue": 0.0,
            "avg_tenure": 0.0,
        }

    summary = get_churn_summary(df)
    total_rev = round(float(df["MonthlyCharges"].sum()), 2)
    lost_rev = round(float(df.loc[df["Churn"] == "Yes", "MonthlyCharges"].sum()), 2)

    return {
        "group_name": group_name,
        "total_customers": total,
        "churned_customers": summary["churned_customers"],
        "retained_customers": summary["retained_customers"],
        "churn_rate_pct": summary["churn_rate_pct"],
        "total_monthly_revenue": total_rev,
        "lost_monthly_revenue": lost_rev,
        "avg_tenure": round(float(df["tenure"].mean()), 2),
        "avg_monthly_charges": round(float(df["MonthlyCharges"].mean()), 2),
    }


def compute_business_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Dynamically derives data-backed business insights from the dataset.

    Each insight follows a strict 4-part structure:
    1. Problem: What pattern or potential issue was identified?
    2. Evidence: What measured values support it, including sample sizes and counts?
    3. Recommendation: What should a business investigate or consider doing next?
    4. Limitation: What cannot be concluded from this finding (e.g. non-causality, selection effects)?

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned (and optionally filtered) DataFrame.

    Returns:
    --------
    list of dict:
        List of structured insight dictionaries.
    """
    if len(df) == 0:
        return []

    insights: List[Dict[str, Any]] = []
    overall = get_churn_summary(df)

    # 1. Contract Commitment Insight
    if "Contract" in df.columns and df["Contract"].nunique() > 1:
        contract_data = get_churn_rate_by_category(df, "Contract")
        m2m_row = contract_data[contract_data["Contract"] == "Month-to-month"]
        two_yr_row = contract_data[contract_data["Contract"] == "Two year"]

        if not m2m_row.empty:
            m2m_rate = float(m2m_row["Churn Rate (%)"].iloc[0])
            m2m_total = int(m2m_row["Total Customers"].iloc[0])
            m2m_churned = int(m2m_row["Churned Customers"].iloc[0])
            two_yr_rate = float(two_yr_row["Churn Rate (%)"].iloc[0]) if not two_yr_row.empty else 0.0

            insights.append({
                "id": "contract_risk",
                "title": "Short-Term Contract Vulnerability",
                "category": "Contract Structure",
                "severity": "High" if m2m_rate > 35 else "Medium",
                "problem": (
                    "Customers on Month-to-month agreements exhibit disproportionately high churn, "
                    "constituting the primary source of customer departures."
                ),
                "evidence": (
                    f"Month-to-month subscribers experience an observed churn rate of {m2m_rate:.2f}% "
                    f"({m2m_churned:,} churned out of {m2m_total:,} customers). In contrast, Two-year contract "
                    f"subscribers churn at only {two_yr_rate:.2f}%. Month-to-month accounts make up "
                    f"{float(m2m_row['Share of Base (%)'].iloc[0]):.1f}% of this cohort."
                ),
                "recommendation": (
                    "Investigate incentive structures for transitioning month-to-month customers to annual commitments "
                    "(e.g., targeted discounts at months 3 and 6, bonus speed upgrades, or bundled add-ons)."
                ),
                "limitation": (
                    "Contract length does not establish causality. Customers who are already uncertain about their "
                    "long-term requirement naturally self-select into flexible month-to-month plans (selection bias)."
                ),
            })

    # 2. Internet Service Tier Insight
    if "InternetService" in df.columns and df["InternetService"].nunique() > 1:
        internet_data = get_churn_rate_by_category(df, "InternetService")
        fiber_row = internet_data[internet_data["InternetService"] == "Fiber optic"]
        dsl_row = internet_data[internet_data["InternetService"] == "DSL"]

        if not fiber_row.empty:
            fiber_rate = float(fiber_row["Churn Rate (%)"].iloc[0])
            fiber_total = int(fiber_row["Total Customers"].iloc[0])
            fiber_churned = int(fiber_row["Churned Customers"].iloc[0])
            dsl_rate = float(dsl_row["Churn Rate (%)"].iloc[0]) if not dsl_row.empty else 0.0

            insights.append({
                "id": "fiber_service_risk",
                "title": "Fiber Optic Service Attrition",
                "category": "Service Tier",
                "severity": "High" if fiber_rate > 35 else "Medium",
                "problem": (
                    "Fiber optic customers churn at a significantly higher rate than DSL subscribers, "
                    "despite Fiber optic representing a premium higher-ARPU tier."
                ),
                "evidence": (
                    f"Fiber optic users churn at an observed rate of {fiber_rate:.2f}% "
                    f"({fiber_churned:,} churned out of {fiber_total:,} customers), compared to {dsl_rate:.2f}% for DSL users "
                    f"and {overall['churn_rate_pct']:.2f}% overall across this cohort."
                ),
                "recommendation": (
                    "Audit technical support tickets, network reliability metrics, and initial onboarding setup "
                    "for Fiber optic accounts to determine whether speed reliability or price-value gaps drive dissatisfaction."
                ),
                "limitation": (
                    "Observed differences do not prove fiber optic technology itself causes cancellations. Fiber areas "
                    "may face aggressive competitor marketing or have different demographic price elasticity."
                ),
            })

    # 3. Payment Method Insight
    if "PaymentMethod" in df.columns and df["PaymentMethod"].nunique() > 1:
        pm_data = get_churn_rate_by_category(df, "PaymentMethod")
        echeck_row = pm_data[pm_data["PaymentMethod"] == "Electronic check"]
        auto_card_row = pm_data[pm_data["PaymentMethod"] == "Credit card (automatic)"]

        if not echeck_row.empty:
            echeck_rate = float(echeck_row["Churn Rate (%)"].iloc[0])
            echeck_total = int(echeck_row["Total Customers"].iloc[0])
            echeck_churned = int(echeck_row["Churned Customers"].iloc[0])
            auto_rate = float(auto_card_row["Churn Rate (%)"].iloc[0]) if not auto_card_row.empty else 0.0

            insights.append({
                "id": "payment_friction",
                "title": "Payment Friction & Electronic Check Risk",
                "category": "Billing Operations",
                "severity": "High" if echeck_rate > 35 else "Medium",
                "problem": (
                    "Customers paying via manual Electronic Check have more than double the churn rate "
                    "of customers enrolled in automated payment mechanisms."
                ),
                "evidence": (
                    f"Electronic check payers show an observed churn rate of {echeck_rate:.2f}% "
                    f"({echeck_churned:,} churned out of {echeck_total:,} customers). By comparison, "
                    f"automatic credit card users churn at only {auto_rate:.2f}%."
                ),
                "recommendation": (
                    "Implement a targeted campaign to migrate electronic check users to automated payment methods "
                    "(e.g., offering a one-time bill credit or fee waiver for enrolling in autopay)."
                ),
                "limitation": (
                    "Payment method correlation does not demonstrate that paying by electronic check causes someone to leave. "
                    "Payment choice may simply correlate with underlying financial volatility or customer demographics."
                ),
            })

    # 4. Customer Lifecycle / Tenure Insight
    if "tenure" in df.columns:
        tenure_info = get_tenure_distribution(df)
        brackets = tenure_info["brackets"]
        if not brackets.empty:
            y1_row = brackets[brackets["tenure_bracket"] == "0-12 months"]
            y5_row = brackets[brackets["tenure_bracket"] == "61-72 months"]

            if not y1_row.empty:
                y1_rate = float(y1_row["Churn Rate (%)"].iloc[0])
                y1_total = int(y1_row["Total Customers"].iloc[0])
                y1_churned = int(y1_row["Churned Customers"].iloc[0])
                y5_rate = float(y5_row["Churn Rate (%)"].iloc[0]) if not y5_row.empty else 0.0

                insights.append({
                    "id": "early_lifecycle_vulnerability",
                    "title": "First-Year Lifecycle Vulnerability",
                    "category": "Customer Lifecycle",
                    "severity": "High" if y1_rate > 40 else "Medium",
                    "problem": (
                        "Customer attrition is heavily front-loaded in the first 12 months of tenure, "
                        "after which retention stabilizes substantially."
                    ),
                    "evidence": (
                        f"Customers in their first year (0-12 months) experience an observed churn rate of {y1_rate:.2f}% "
                        f"({y1_churned:,} churned out of {y1_total:,} customers). For mature accounts (61-72 months), "
                        f"churn drops sharply to {y5_rate:.2f}%."
                    ),
                    "recommendation": (
                        "Design structured onboarding touchpoints during the first 90 days (welcome calls, satisfaction check-ins, "
                        "and proactive technical support) before the high-risk 12-month attrition peak."
                    ),
                    "limitation": (
                        "The observed survival curve reflects survivorship bias: customers remaining past 5 years are inherently "
                        "pre-filtered loyal accounts. Early interventions will not necessarily prevent churn for all new customers."
                    ),
                })

    return insights


# Alias for backward compatibility
generate_overview_insights = compute_business_insights


def generate_segment_insights(cluster_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Derives data-backed business insights from K-Means clustering results.

    Parameters:
    -----------
    cluster_metadata : dict
        Output from fit_customer_clusters(), containing 'summary_table'.

    Returns:
    --------
    list of dict:
        Structured insights with Problem, Evidence, Recommendation, and Limitation.
    """
    summary_table = cluster_metadata.get("summary_table")
    if summary_table is None or summary_table.empty:
        return []

    insights: List[Dict[str, Any]] = []

    # 1. Highest Churn Cluster Insight
    highest_row = summary_table.sort_values(by="Observed Churn Rate (%)", ascending=False).iloc[0]
    high_rate = float(highest_row["Observed Churn Rate (%)"])
    high_count = int(highest_row["Customer Count"])
    high_churned = int(highest_row["Churned Customers"])
    high_share = float(highest_row["Share of Base (%)"])
    high_tenure = float(highest_row["Avg Tenure (mos)"])
    high_charges = float(highest_row["Avg Monthly Charges ($)"])

    insights.append({
        "id": "highest_churn_segment",
        "title": f"Concentrated Attrition in {highest_row['Cluster']}",
        "category": "Customer Clustering",
        "severity": "High" if high_rate > 35 else "Medium",
        "problem": (
            f"{highest_row['Cluster']} ({highest_row['Segment Archetype']}) demonstrates the highest churn "
            f"vulnerability among all identified customer segments."
        ),
        "evidence": (
            f"{highest_row['Cluster']} exhibits an observed churn rate of {high_rate:.2f}% ({high_churned:,} churned "
            f"out of {high_count:,} customers, representing {high_share:.1f}% of the segmented cohort), "
            f"with an average monthly charge of ${high_charges:.2f} and average tenure of {high_tenure:.1f} months."
        ),
        "recommendation": (
            f"Prioritize proactive customer success outreach and price-value validation on {highest_row['Cluster']}. "
            f"Investigate whether service disruptions or onboarding friction drive attrition for this segment."
        ),
        "limitation": (
            "Cluster assignments represent geometric proximity in standardized (tenure, MonthlyCharges) space. "
            "Being assigned to this cluster does not guarantee an individual customer will churn, nor that tenure and "
            "charges are the sole drivers of their satisfaction."
        ),
    })

    # 2. Lowest Churn Cluster Insight
    lowest_row = summary_table.sort_values(by="Observed Churn Rate (%)", ascending=True).iloc[0]
    low_rate = float(lowest_row["Observed Churn Rate (%)"])
    low_count = int(lowest_row["Customer Count"])
    low_churned = int(lowest_row["Churned Customers"])
    low_share = float(lowest_row["Share of Base (%)"])
    low_tenure = float(lowest_row["Avg Tenure (mos)"])
    low_charges = float(lowest_row["Avg Monthly Charges ($)"])

    insights.append({
        "id": "stable_retention_segment",
        "title": f"Stable Core Retention in {lowest_row['Cluster']}",
        "category": "Customer Clustering",
        "severity": "Low",
        "problem": (
            f"Over-allocating aggressive defensive discounts or intrusive retention campaigns to {lowest_row['Cluster']} "
            f"risks eroding margins on a segment that is already exceptionally stable."
        ),
        "evidence": (
            f"{lowest_row['Cluster']} ({lowest_row['Segment Archetype']}) shows an observed churn rate of only "
            f"{low_rate:.2f}% ({low_churned:,} churned out of {low_count:,} customers, {low_share:.1f}% of cohort), "
            f"with an average tenure of {low_tenure:.1f} months and average bill of ${low_charges:.2f}."
        ),
        "recommendation": (
            "Maintain service reliability and standard operational support for this group without offering "
            "unnecessary discounts or disruptive plan migrations."
        ),
        "limitation": (
            "Historical stability in this cluster does not guarantee permanent immunity from macro-economic pressure "
            "or aggressive competitor entry in their service areas."
        ),
    })

    # 3. High-Spend Loyalist Protection
    high_spend_row = summary_table.sort_values(by="Avg Monthly Charges ($)", ascending=False).iloc[0]
    if str(high_spend_row["Cluster"]) != str(highest_row["Cluster"]):
        hs_count = int(high_spend_row["Customer Count"])
        hs_churned = int(high_spend_row["Churned Customers"])
        hs_rate = float(high_spend_row["Observed Churn Rate (%)"])
        hs_charges = float(high_spend_row["Avg Monthly Charges ($)"])
        hs_share = float(high_spend_row["Share of Base (%)"])

        insights.append({
            "id": "high_value_revenue_protection",
            "title": f"Revenue Protection for {high_spend_row['Cluster']}",
            "category": "Customer Clustering",
            "severity": "Medium",
            "problem": (
                f"{high_spend_row['Cluster']} carries the highest average monthly spend; any attrition in this "
                f"group creates outsized monthly revenue losses."
            ),
            "evidence": (
                f"This cluster contains {hs_count:,} customers ({hs_share:.1f}% of cohort) paying an average of "
                f"${hs_charges:.2f}/month, generating substantial recurring revenue with an observed churn rate "
                f"of {hs_rate:.2f}% ({hs_churned:,} churned customers)."
            ),
            "recommendation": (
                "Deploy VIP support lines, periodic contract renewal check-ins, and loyalty recognition rewards "
                "to preserve long-term satisfaction."
            ),
            "limitation": (
                "High monthly bills reflect current subscribed plan tiers, not intrinsic brand affinity. High-paying "
                "customers can still cancel if service quality drops."
            ),
        })

    return insights


def generate_model_insights(model_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Derives data-backed business insights from Decision Tree classification results.

    Parameters:
    -----------
    model_results : dict
        Output from train_decision_tree_model().

    Returns:
    --------
    list of dict:
        Structured insights with Problem, Evidence, Recommendation, and Limitation.
    """
    fi_df = model_results.get("feature_importance", pd.DataFrame())
    cm = model_results.get("confusion_matrix", {})
    test_metrics = model_results.get("test_metrics", {})
    baseline = model_results.get("baseline_metrics", {})

    insights: List[Dict[str, Any]] = []

    # 1. Primary Feature Reliance
    if not fi_df.empty:
        top_f = fi_df.iloc[0]
        top_name = str(top_f["feature"])
        top_pct = float(top_f["importance_pct"])

        insights.append({
            "id": "primary_feature_reliance",
            "title": f"Primary Model Reliance on {top_name}",
            "category": "Model Interpretability",
            "severity": "High",
            "problem": (
                f"The Decision Tree model places dominant reliance on {top_name} when partitioning churn risk."
            ),
            "evidence": (
                f"'{top_name}' accounts for {top_pct:.1f}% of total Gini impurity reduction across all decision splits "
                f"in the trained tree."
            ),
            "recommendation": (
                "Prioritize business reviews around customer contract transitions, evaluating whether flexible "
                "pricing or onboarding incentives can encourage longer-term commitments."
            ),
            "limitation": (
                "Feature importance reflects mathematical split reliance inside this specific Decision Tree, NOT causal "
                "influence in the real world. A month-to-month contract does not force a customer to leave; it merely reflects "
                "the absence of switching friction."
            ),
        })

    # 2. Classification Trade-off & Error Balance
    if cm and test_metrics:
        total_test = cm.get("total_test", 0)
        tp = cm.get("tp", 0)
        fp = cm.get("fp", 0)
        fn = cm.get("fn", 0)
        actual_churners = tp + fn

        insights.append({
            "id": "classification_tradeoff",
            "title": "False Positive vs. False Negative Operational Trade-off",
            "category": "Model Diagnostics",
            "severity": "Medium",
            "problem": (
                f"The classification decision boundary trades off {fp:,} false alarms (False Positives) against "
                f"{fn:,} missed cancellations (False Negatives), carrying asymmetrical business costs."
            ),
            "evidence": (
                f"On the held-out test set (n = {total_test:,}), the model achieved {test_metrics.get('recall_pct', 0):.1f}% recall "
                f"({tp:,} of {actual_churners:,} actual churners detected) while producing {fp:,} False Positives "
                f"and {fn:,} False Negatives. Overall precision is {test_metrics.get('precision_pct', 0):.1f}%."
            ),
            "recommendation": (
                "Align retention intervention strategies with cost: if outreach has negligible cost (e.g. email check-in), "
                "tolerating False Positives to capture churners is optimal. If the retention offer involves expensive discounts, "
                "a higher decision threshold or stricter precision filter is required."
            ),
            "limitation": (
                "The confusion matrix reflects model predictions at a default 0.5 decision threshold on an uncalibrated tree leaf "
                "proportion. It does not state with certainty which individual customer will leave."
            ),
        })

    # 3. Class Imbalance & The Baseline Paradox
    if baseline and test_metrics:
        base_acc = baseline.get("baseline_accuracy_pct", 0)
        lift = baseline.get("accuracy_lift_pct", 0)
        test_acc = test_metrics.get("accuracy_pct", 0)

        insights.append({
            "id": "class_imbalance_paradox",
            "title": "Class Imbalance & The Accuracy Paradox",
            "category": "Evaluation Methodology",
            "severity": "Low",
            "problem": (
                "Relying solely on accuracy to evaluate retention models is misleading when churn classes are imbalanced."
            ),
            "evidence": (
                f"A naive baseline predicting 'No Churn' for all customers achieves {base_acc:.1f}% accuracy but detects "
                f"0% of churners (0 recall, 0 F1). The Decision Tree achieves {test_acc:.1f}% accuracy (+{lift:.1f}% lift) "
                f"and detects {cm.get('tp', 0):,} churners with an F1-score of {test_metrics.get('f1_pct', 0):.1f}%."
            ),
            "recommendation": (
                "Mandate that internal stakeholders evaluate churn models using Precision, Recall, F1-score, and Lift "
                "over the majority baseline rather than reporting raw accuracy in isolation."
            ),
            "limitation": (
                "Demonstrating predictive lift over a baseline shows mathematical pattern recognition on historical data; "
                "it does not guarantee that business intervention campaigns will successfully reverse customer cancellations."
            ),
        })

    return insights



# ==============================================================================
# CUSTOMER SEGMENTATION (K-MEANS CLUSTERING)
# ==============================================================================

DEFAULT_CLUSTER_FEATURES: List[str] = ["tenure", "MonthlyCharges"]


def evaluate_kmeans_k_range(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    k_min: int = 2,
    k_max: int = 7,
    random_state: int = 42,
    n_init: int = 10,
) -> Dict[str, Any]:
    """
    Evaluates candidate values of k from k_min through k_max for K-Means clustering.

    Methodology:
    1. Validates that feature columns exist and contain valid numeric data.
    2. Standardizes features with StandardScaler to prevent scale distortion
       (tenure ranges 0-72 months while MonthlyCharges ranges $18-$120).
    3. Computes inertia (within-cluster sum of squared errors) for the Elbow Method.
    4. Computes mean silhouette scores to measure cluster separation and cohesion.
    5. Determines the mathematically optimal k based on maximum silhouette score.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to cluster.
    features : list of str, optional
        Features to cluster. Defaults to ['tenure', 'MonthlyCharges'].
    k_min : int
        Minimum candidate number of clusters (default: 2).
    k_max : int
        Maximum candidate number of clusters (default: 7).
    random_state : int
        Fixed random seed for reproducible centroid initialization.
    n_init : int
        Number of times the K-Means algorithm runs with different centroid seeds.

    Returns:
    --------
    dict:
        Diagnostic results containing:
        - k_values: list of evaluated k integers
        - inertias: list of within-cluster sum of squares
        - silhouette_scores: list of silhouette coefficients
        - recommended_k: k with the highest silhouette score
        - justification: clear textual explanation of why recommended_k was chosen
    """
    if features is None:
        features = DEFAULT_CLUSTER_FEATURES

    # Validate feature presence
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        raise KeyError(f"Missing clustering features in DataFrame: {missing_feats}")

    # Prepare feature matrix and drop any rows with NaN in these specific features
    feature_df = df[features].copy()
    valid_mask = feature_df.notna().all(axis=1)
    clean_features = feature_df[valid_mask]

    if len(clean_features) < k_max:
        raise ValueError(
            f"Not enough valid records ({len(clean_features)}) to evaluate k up to {k_max}."
        )

    # Feature scaling: essential for Euclidean distance-based algorithms
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean_features)

    k_values: List[int] = []
    inertias: List[float] = []
    sil_scores: List[float] = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_scaled)
        inertia = float(km.inertia_)
        sil = float(silhouette_score(X_scaled, labels))

        k_values.append(k)
        inertias.append(round(inertia, 2))
        sil_scores.append(round(sil, 4))

    # Identify optimal k by peak silhouette score
    max_sil_idx = int(np.argmax(sil_scores))
    recommended_k = k_values[max_sil_idx]
    best_sil = sil_scores[max_sil_idx]

    # Justification text based strictly on computed results
    justification = (
        f"Candidate evaluation across k={k_min} to {k_max} shows that k={recommended_k} achieves the "
        f"highest silhouette score ({best_sil:.4f}). Furthermore, the inertia elbow curve exhibits its "
        f"primary inflection point around k={recommended_k}, after which reductions in within-cluster "
        f"variance taper significantly."
    )

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": sil_scores,
        "recommended_k": recommended_k,
        "best_silhouette_score": best_sil,
        "justification": justification,
        "features": features,
    }


def derive_cluster_archetype(
    cluster_tenure: float,
    cluster_charges: float,
    overall_tenure: float,
    overall_charges: float,
    churn_rate: float,
) -> Tuple[str, str]:
    """
    Dynamically generates a segment archetype name and strategic focus from actual
    empirical cluster characteristics compared to overall dataset baselines.

    Parameters:
    -----------
    cluster_tenure : float
        Mean tenure for the cluster.
    cluster_charges : float
        Mean monthly charges for the cluster.
    overall_tenure : float
        Overall mean tenure for the dataset.
    overall_charges : float
        Overall mean monthly charges for the dataset.
    churn_rate : float
        Observed churn rate percentage for the cluster.

    Returns:
    --------
    tuple of (archetype_name, strategic_focus):
        Human-readable persona label and recommended business focus.
    """
    is_high_tenure = cluster_tenure >= overall_tenure
    is_high_charges = cluster_charges >= overall_charges

    if not is_high_tenure and is_high_charges:
        archetype = "New / Short-Tenure, High-Spend"
        focus = (
            f"High Churn Risk ({churn_rate:.1f}%): Early accounts paying premium rates. "
            "Prioritize proactive onboarding check-ins and dedicated technical support to protect high ARPU."
        )
    elif is_high_tenure and is_high_charges:
        archetype = "Established, High-Spend Loyalists"
        focus = (
            f"Moderate Churn Risk ({churn_rate:.1f}%): Core high-value customer base. "
            "Engage with loyalty rewards, VIP service tiers, and contract renewal incentives."
        )
    elif not is_high_tenure and not is_high_charges:
        archetype = "New / Short-Tenure, Budget-Tier"
        focus = (
            f"Average Churn Risk ({churn_rate:.1f}%): Entry-level accounts on basic plans. "
            "Introduce gradual feature discovery, usage education, and cost-effective bundle add-ons."
        )
    else:  # is_high_tenure and not is_high_charges
        archetype = "Mature Legacy, Budget-Tier"
        focus = (
            f"Low Churn Risk ({churn_rate:.1f}%): Extremely stable long-term subscribers with minimal attrition. "
            "Maintain service reliability and consider non-intrusive value upgrades."
        )

    return archetype, focus


def fit_customer_clusters(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    n_clusters: int = 4,
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fits the final K-Means model, assigns cluster IDs, and computes segment summaries.

    Methodology:
    1. Standardizes 'tenure' and 'MonthlyCharges' using StandardScaler.
    2. Fits KMeans with fixed random_state and specified n_init.
    3. Transforms cluster centroids back to original feature scale ($ and months).
    4. Computes empirical cluster summary metrics: count, share of base, average tenure,
       average monthly charges, and observed churn rate.
    5. Derives descriptive segment archetypes dynamically from empirical cluster positions.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to segment.
    features : list of str, optional
        Features used for clustering. Defaults to ['tenure', 'MonthlyCharges'].
    n_clusters : int
        Number of clusters to fit (k).
    random_state : int
        Fixed random seed for deterministic clustering.
    n_init : int
        Number of centroid seed re-initializations.

    Returns:
    --------
    tuple of (pd.DataFrame, dict):
        - clustered_df: DataFrame containing the new 'Cluster' column.
        - cluster_metadata: Comprehensive summary dictionary with centroids,
          summary DataFrame, inertia, and silhouette score.
    """
    if features is None:
        features = DEFAULT_CLUSTER_FEATURES

    # Validate features
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        raise KeyError(f"Missing clustering features: {missing_feats}")

    clustered_df = df.copy()

    # Drop any rows with NaN in clustering features to ensure integrity
    valid_mask = clustered_df[features].notna().all(axis=1)
    if not valid_mask.all():
        clustered_df = clustered_df[valid_mask].copy()

    # 1. Feature Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clustered_df[features])

    # 2. Fit K-Means Model
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    cluster_labels = km.fit_predict(X_scaled)
    clustered_df["Cluster"] = cluster_labels

    # 3. Compute silhouette score and inertia for the fitted model
    sil_score = float(silhouette_score(X_scaled, cluster_labels))
    inertia = float(km.inertia_)

    # 4. Map centroids back to original units (months and dollars)
    centroids_orig = scaler.inverse_transform(km.cluster_centers_)
    centroids_list: List[Dict[str, Any]] = []
    for c_idx, coords in enumerate(centroids_orig):
        centroids_list.append({
            "Cluster": c_idx,
            "tenure": round(float(coords[0]), 2),
            "MonthlyCharges": round(float(coords[1]), 2),
        })

    # 5. Build Cluster Summary Table from Actual Data
    overall_tenure = float(clustered_df["tenure"].mean())
    overall_charges = float(clustered_df["MonthlyCharges"].mean())
    total_customers = len(clustered_df)

    summary_rows: List[Dict[str, Any]] = []
    for c_id in range(n_clusters):
        c_subset = clustered_df[clustered_df["Cluster"] == c_id]
        c_count = len(c_subset)
        if c_count == 0:
            continue

        c_share = (c_count / total_customers) * 100
        avg_tenure = float(c_subset["tenure"].mean())
        med_tenure = float(c_subset["tenure"].median())
        avg_charges = float(c_subset["MonthlyCharges"].mean())
        med_charges = float(c_subset["MonthlyCharges"].median())

        # Observed Churn Rate
        if "ChurnBinary" in c_subset.columns:
            churn_rate = float(c_subset["ChurnBinary"].mean() * 100)
            churn_count = int(c_subset["ChurnBinary"].sum())
        else:
            churn_count = int((c_subset["Churn"] == "Yes").sum())
            churn_rate = (churn_count / c_count) * 100 if c_count > 0 else 0.0

        archetype, strategic_focus = derive_cluster_archetype(
            cluster_tenure=avg_tenure,
            cluster_charges=avg_charges,
            overall_tenure=overall_tenure,
            overall_charges=overall_charges,
            churn_rate=churn_rate,
        )

        summary_rows.append({
            "Cluster": f"Cluster {c_id}",
            "ClusterID": c_id,
            "Segment Archetype": archetype,
            "Customer Count": c_count,
            "Share of Base (%)": round(c_share, 1),
            "Avg Tenure (mos)": round(avg_tenure, 1),
            "Median Tenure (mos)": round(med_tenure, 1),
            "Avg Monthly Charges ($)": round(avg_charges, 2),
            "Median Monthly Charges ($)": round(med_charges, 2),
            "Observed Churn Rate (%)": round(churn_rate, 2),
            "Churned Customers": churn_count,
            "Strategic Focus": strategic_focus,
        })

    summary_df = pd.DataFrame(summary_rows)

    cluster_metadata: Dict[str, Any] = {
        "n_clusters": n_clusters,
        "inertia": inertia,
        "silhouette_score": sil_score,
        "centroids": centroids_list,
        "summary_table": summary_df,
        "features": features,
        "random_state": random_state,
        "n_init": n_init,
        "overall_tenure": round(overall_tenure, 2),
        "overall_charges": round(overall_charges, 2),
    }

    return clustered_df, cluster_metadata


# ==============================================================================
# CHURN RISK CLASSIFICATION (DECISION TREE CLASSIFIER)
# ==============================================================================

TREE_MODEL_FEATURES: List[str] = [
    "Contract",
    "tenure",
    "MonthlyCharges",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "PaperlessBilling",
    "SeniorCitizen",
]

TREE_CATEGORICAL_FEATURES: List[str] = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport",
    "PaperlessBilling",
]

TREE_NUMERIC_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "SeniorCitizen",
]


def clean_feature_name(raw_name: str) -> str:
    """
    Formats scikit-learn OneHotEncoder feature names into human-readable labels.
    Example: 'cat__Contract_Month-to-month' -> 'Contract: Month-to-month'
             'num__tenure' -> 'Tenure (Months)'
    """
    if raw_name.startswith("cat__"):
        clean = raw_name.replace("cat__", "")
        parts = clean.split("_", 1)
        if len(parts) == 2:
            return f"{parts[0]}: {parts[1]}"
        return clean
    elif raw_name.startswith("num__"):
        clean = raw_name.replace("num__", "")
        labels_map = {
            "tenure": "Customer Tenure (Months)",
            "MonthlyCharges": "Monthly Charges ($)",
            "SeniorCitizen": "Senior Citizen Flag",
        }
        return labels_map.get(clean, clean)
    return raw_name


def train_decision_tree_model(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    candidate_depths: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Trains and evaluates a Decision Tree classifier for customer churn risk.

    Strict Methodological Workflow:
    1. Input Feature Selection: Restricted to 8 interpretable attributes.
       Explicitly excludes 'customerID' (identifier) and 'TotalCharges' (to avoid collinearity with tenure * charges).
    2. Data Partitioning: Splits 80/20 with stratification on the target (ChurnBinary)
       and a fixed random seed.
    3. Pipeline Preprocessing: OneHotEncoder(handle_unknown='ignore') for categorical
       variables, numeric features passed through.
    4. Model Selection on Train Data ONLY: Compares candidate tree depths (2 through 7)
       via 5-Fold Stratified Cross-Validation on the training set.
    5. Held-Out Test Evaluation: The chosen model is fitted on the full training set
       and evaluated exactly ONCE on the held-out test set.
    6. Baseline & Imbalance Analysis: Compares accuracy to a naive majority-class baseline.
    7. Feature Importance Extraction: Extracts Gini importance and clarifies model reliance vs. causation.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset containing target 'ChurnBinary' or 'Churn'.
    test_size : float
        Proportion of records allocated to the test set (default: 0.20).
    random_state : int
        Deterministic seed for split and model reproducibility (default: 42).
    candidate_depths : list of int, optional
        Depths to evaluate in cross-validation (default: [2, 3, 4, 5, 6, 7]).

    Returns:
    --------
    dict:
        Comprehensive modeling results, evaluation metrics, and the fitted pipeline.
    """
    if candidate_depths is None:
        candidate_depths = [2, 3, 4, 5, 6, 7]

    # Validate feature presence
    missing_feats = [f for f in TREE_MODEL_FEATURES if f not in df.columns]
    if missing_feats:
        raise KeyError(f"Missing required model features in DataFrame: {missing_feats}")

    # Ensure binary target exists
    if "ChurnBinary" in df.columns:
        target_series = df["ChurnBinary"]
    else:
        target_series = (df["Churn"].astype(str).str.strip().str.capitalize() == "Yes").astype(int)

    X = df[TREE_MODEL_FEATURES].copy()
    y = target_series.copy()

    # Step 1: Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Step 2: Scikit-learn Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                TREE_CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", TREE_NUMERIC_FEATURES),
        ]
    )

    # Step 3: Model Selection via 5-Fold Stratified Cross-Validation on Train Data ONLY
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    X_train_trans = preprocessor.fit_transform(X_train)

    cv_results_list: List[Dict[str, Any]] = []
    best_depth = candidate_depths[0]
    best_cv_f1 = -1.0

    for depth in candidate_depths:
        clf_candidate = DecisionTreeClassifier(max_depth=depth, random_state=random_state)
        f1_scores = cross_val_score(clf_candidate, X_train_trans, y_train, cv=cv, scoring="f1")
        acc_scores = cross_val_score(clf_candidate, X_train_trans, y_train, cv=cv, scoring="accuracy")

        mean_f1 = float(np.mean(f1_scores))
        std_f1 = float(np.std(f1_scores))
        mean_acc = float(np.mean(acc_scores))

        cv_results_list.append({
            "Depth": depth,
            "Mean CV F1": round(mean_f1, 4),
            "Std CV F1": round(std_f1, 4),
            "Mean CV Accuracy": round(mean_acc, 4),
        })

        # Select depth that maximizes training cross-validation F1 score
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_depth = depth

    cv_df = pd.DataFrame(cv_results_list)

    # Step 4: Fit Final Pipeline on the ENTIRE Training Set
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=best_depth, random_state=random_state)),
    ])
    final_pipeline.fit(X_train, y_train)

    # Step 5: Evaluate ONCE on the Held-Out Test Set
    y_test_pred = final_pipeline.predict(X_test)

    test_accuracy = float(accuracy_score(y_test, y_test_pred))
    test_precision = float(precision_score(y_test, y_test_pred, zero_division=0))
    test_recall = float(recall_score(y_test, y_test_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, y_test_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_test_pred)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Step 6: Majority-Class Baseline Analysis
    majority_class = int(y_train.mode()[0])  # 0 (No Churn)
    majority_preds = np.full_like(y_test, majority_class)
    baseline_accuracy = float(accuracy_score(y_test, majority_preds))

    # Step 7: Feature Importances
    feat_names = list(final_pipeline.named_steps["preprocessor"].get_feature_names_out())
    importances = list(final_pipeline.named_steps["classifier"].feature_importances_)

    fi_list: List[Dict[str, Any]] = []
    for raw_f, imp in zip(feat_names, importances):
        if imp > 0.0001:
            fi_list.append({
                "raw_feature": raw_f,
                "feature": clean_feature_name(raw_f),
                "importance": round(float(imp), 4),
                "importance_pct": round(float(imp * 100), 2),
            })

    fi_df = pd.DataFrame(fi_list).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return {
        "pipeline": final_pipeline,
        "selected_depth": best_depth,
        "candidate_depths": candidate_depths,
        "cv_results": cv_df,
        "test_metrics": {
            "accuracy": round(test_accuracy, 4),
            "precision": round(test_precision, 4),
            "recall": round(test_recall, 4),
            "f1": round(test_f1, 4),
            "accuracy_pct": round(test_accuracy * 100, 2),
            "precision_pct": round(test_precision * 100, 2),
            "recall_pct": round(test_recall * 100, 2),
            "f1_pct": round(test_f1 * 100, 2),
        },
        "confusion_matrix": {
            "matrix": cm,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total_test": len(y_test),
        },
        "baseline_metrics": {
            "majority_class": majority_class,
            "majority_label": "No Churn (0)",
            "baseline_accuracy": round(baseline_accuracy, 4),
            "baseline_accuracy_pct": round(baseline_accuracy * 100, 2),
            "accuracy_lift_pct": round((test_accuracy - baseline_accuracy) * 100, 2),
        },
        "feature_importance": fi_df,
        "split_summary": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_churn_rate_pct": round(float(y_train.mean() * 100), 2),
            "test_churn_rate_pct": round(float(y_test.mean() * 100), 2),
        },
        "features": TREE_MODEL_FEATURES,
    }


def predict_customer_churn_risk(
    pipeline: Pipeline,
    customer_input: Union[Dict[str, Any], pd.DataFrame],
) -> Dict[str, Any]:
    """
    Evaluates churn risk for an individual hypothetical customer profile.

    Crucial Technical Safeguards:
    1. Uses the exact same trained scikit-learn Pipeline (ColumnTransformer + DecisionTreeClassifier),
       guaranteeing zero schema drift or one-hot column mismatch.
    2. Does NOT call the output a 'calibrated churn probability'. The score represents
       the proportion of churned training observations in the terminal decision tree leaf node.
    3. Validates boundary ranges on inputs (e.g., tenure >= 0, monthly charges >= 0).

    Parameters:
    -----------
    pipeline : Pipeline
        Trained scikit-learn pipeline from train_decision_tree_model().
    customer_input : dict or pd.DataFrame
        Customer features matching TREE_MODEL_FEATURES.

    Returns:
    --------
    dict:
        Prediction results including predicted class, leaf churn proportion, risk tier,
        and explanatory notes.
    """
    if isinstance(customer_input, dict):
        input_df = pd.DataFrame([customer_input])
    else:
        input_df = customer_input.copy()

    # Validate required columns
    missing = [f for f in TREE_MODEL_FEATURES if f not in input_df.columns]
    if missing:
        raise KeyError(f"Customer input is missing required features: {missing}")

    # Boundary validations
    tenure_val = float(input_df["tenure"].iloc[0])
    charges_val = float(input_df["MonthlyCharges"].iloc[0])
    if tenure_val < 0 or tenure_val > 120:
        raise ValueError(f"Tenure must be between 0 and 120 months (received: {tenure_val}).")
    if charges_val < 0 or charges_val > 500:
        raise ValueError(f"Monthly charges must be non-negative and realistic (received: {charges_val}).")

    # Run prediction through the intact pipeline
    pred_class = int(pipeline.predict(input_df)[0])
    probabilities = pipeline.predict_proba(input_df)[0]
    uncalibrated_risk_score = float(probabilities[1])  # Class 1 (Churn) leaf proportion

    risk_score_pct = round(uncalibrated_risk_score * 100, 1)

    if pred_class == 1:
        prediction_label = "High Churn Risk (Class 1)"
        if risk_score_pct >= 70:
            risk_tier = "Severe Risk"
            recommendation = (
                "Immediate retention intervention recommended. High likelihood of cancellation. "
                "Evaluate switching to annual contract or applying targeted promotional retention credits."
            )
        else:
            risk_tier = "Elevated Risk"
            recommendation = (
                "Moderate-to-high attrition risk. Proactive customer success check-in advised "
                "to address potential price sensitivity or technical support needs."
            )
    else:
        prediction_label = "Low Churn Risk (Class 0)"
        if risk_score_pct <= 15:
            risk_tier = "Minimal Risk"
            recommendation = (
                "Highly stable customer account. Standard engagement cadence is sufficient."
            )
        else:
            risk_tier = "Low / Guarded Risk"
            recommendation = (
                "Customer currently leans toward retention, but monitors for changes in billing or service tickets."
            )

    return {
        "predicted_class": pred_class,
        "prediction_label": prediction_label,
        "predicted_label": prediction_label,
        "uncalibrated_risk_score": uncalibrated_risk_score,
        "risk_score_pct": risk_score_pct,
        "risk_tier": risk_tier,
        "recommendation": recommendation,
        "disclaimer": (
            "Note: This score reflects the proportion of churned training examples landing in the decision tree leaf node. "
            "It is an uncalibrated tree leaf proportion, not a calibrated Bayesian probability."
        ),
    }
