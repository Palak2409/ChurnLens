"""
Automated unit tests for churnlens.py
Tests data loading, validation, cleaning, and analytical calculations.
"""

import pandas as pd
import pytest

from churnlens import (
    REQUIRED_COLUMNS,
    TREE_MODEL_FEATURES,
    check_data_quality,
    clean_data,
    compute_business_insights,
    evaluate_kmeans_k_range,
    filter_customers,
    fit_customer_clusters,
    generate_model_insights,
    generate_segment_insights,
    get_charge_and_tenure_metrics,
    get_churn_rate_by_category,
    get_churn_summary,
    get_group_churn_metrics,
    get_tenure_distribution,
    load_data,
    predict_customer_churn_risk,
    train_decision_tree_model,
    validate_columns,
)
from visualizations import (
    build_insight_card_html,
    create_category_churn_chart,
    create_cluster_scatter_plot,
    create_confusion_matrix_chart,
    create_depth_cv_chart,
    create_elbow_and_silhouette_chart,
    create_feature_importance_chart,
    create_tenure_distribution_chart,
)


@pytest.fixture
def raw_dataset() -> pd.DataFrame:
    """Fixture that loads the actual project dataset."""
    return load_data()


@pytest.fixture
def cleaned_dataset(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    """Fixture that provides the cleaned dataset."""
    cleaned_df, _ = clean_data(raw_dataset)
    return cleaned_df


@pytest.fixture
def synthetic_sample_df() -> pd.DataFrame:
    """Fixture providing a controlled small synthetic dataset."""
    data = {
        "customerID": ["C001", "C002", "C003", "C004", "C005"],
        "gender": ["Female", "Male", "Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "No", "No"],
        "tenure": [1, 0, 24, 0, 12],
        "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes"],
        "MultipleLines": ["No", "No", "No phone service", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic"],
        "OnlineSecurity": ["No", "No", "Yes", "No internet service", "No"],
        "OnlineBackup": ["Yes", "No", "No", "No internet service", "Yes"],
        "DeviceProtection": ["No", "No", "Yes", "No internet service", "No"],
        "TechSupport": ["No", "No", "Yes", "No internet service", "No"],
        "StreamingTV": ["No", "Yes", "No", "No internet service", "Yes"],
        "StreamingMovies": ["No", "Yes", "No", "No internet service", "Yes"],
        "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "Yes", "No", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Electronic check", "Mailed check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 80.0, 45.0, 20.0, 90.0],
        "TotalCharges": ["29.85", " ", "1080.0", " ", "1080.0"],
        "Churn": ["No", "Yes", "No", "No", "Yes"],
    }
    return pd.DataFrame(data)


# ------------------------------------------------------------------------------
# Loading & Validation Tests
# ------------------------------------------------------------------------------

def test_load_data_returns_dataframe(raw_dataset: pd.DataFrame):
    """Test that load_data() successfully returns a non-empty DataFrame."""
    assert isinstance(raw_dataset, pd.DataFrame)
    assert not raw_dataset.empty
    assert len(raw_dataset) > 1000  # Should contain thousands of records


def test_load_data_invalid_path():
    """Test that load_data() raises FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_folder/fake_file.csv")


def test_validate_columns(raw_dataset: pd.DataFrame):
    """Test that required columns are recognized and verified."""
    is_valid, missing = validate_columns(raw_dataset)
    assert is_valid is True
    assert len(missing) == 0

    # Test with intentionally missing column
    corrupted = raw_dataset.drop(columns=["Churn"])
    is_valid_corr, missing_corr = validate_columns(corrupted)
    assert is_valid_corr is False
    assert "Churn" in missing_corr


def test_check_data_quality(raw_dataset: pd.DataFrame):
    """Test that the quality check accurately detects shape and blank strings."""
    report = check_data_quality(raw_dataset)
    assert report["num_rows"] == len(raw_dataset)
    assert report["num_cols"] == len(raw_dataset.columns)
    # The raw Telco dataset should have exactly 11 blank strings in TotalCharges
    assert report["whitespace_counts"].get("TotalCharges", 0) == 11
    assert report["duplicate_customer_ids"] == 0


# ------------------------------------------------------------------------------
# Cleaning Tests
# ------------------------------------------------------------------------------

def test_clean_data_preserves_zero_tenure(synthetic_sample_df: pd.DataFrame):
    """
    Critical requirement:
    Verify that blank TotalCharges with tenure == 0 are safely imputed to 0.0
    and that no records are dropped unnecessarily.
    """
    initial_len = len(synthetic_sample_df)
    cleaned, report = clean_data(synthetic_sample_df)

    assert len(cleaned) == initial_len, "No records should be dropped for blank TotalCharges"
    assert report["zero_tenure_customers_preserved"] == 2
    assert report["blank_total_charges_found"] == 2

    # Check that customers with tenure == 0 have TotalCharges == 0.0
    zero_tenure_tc = cleaned.loc[cleaned["tenure"] == 0, "TotalCharges"]
    assert (zero_tenure_tc == 0.0).all()
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])


def test_clean_data_removes_duplicates():
    """Verify duplicate customerID records are removed and reported."""
    df_with_dups = pd.DataFrame({
        col: ["test"] * 3 for col in REQUIRED_COLUMNS
    })
    df_with_dups["customerID"] = ["A1", "A1", "B2"]
    df_with_dups["tenure"] = [5, 10, 15]
    df_with_dups["MonthlyCharges"] = [20.0, 20.0, 30.0]
    df_with_dups["TotalCharges"] = ["100.0", "200.0", "450.0"]
    df_with_dups["SeniorCitizen"] = [0, 0, 1]
    df_with_dups["Churn"] = ["No", "Yes", "No"]

    cleaned, report = clean_data(df_with_dups)
    assert report["duplicates_removed"] == 1
    assert "A1" in report["duplicate_ids_removed"]
    assert len(cleaned) == 2
    assert cleaned["customerID"].tolist() == ["A1", "B2"]
    # Retains first occurrence (tenure = 5)
    assert cleaned.loc[cleaned["customerID"] == "A1", "tenure"].iloc[0] == 5


def test_clean_data_target_encoding(synthetic_sample_df: pd.DataFrame):
    """Verify ChurnBinary is created correctly while preserving Churn Yes/No."""
    cleaned, _ = clean_data(synthetic_sample_df)
    assert "ChurnBinary" in cleaned.columns
    assert "Churn" in cleaned.columns

    # Verify 1:1 mapping
    assert (cleaned.loc[cleaned["Churn"] == "Yes", "ChurnBinary"] == 1).all()
    assert (cleaned.loc[cleaned["Churn"] == "No", "ChurnBinary"] == 0).all()
    assert set(cleaned["Churn"].unique()).issubset({"Yes", "No"})
    assert set(cleaned["ChurnBinary"].unique()).issubset({0, 1})


def test_clean_data_on_actual_dataset(raw_dataset: pd.DataFrame):
    """Test cleaning on the real Telco dataset."""
    cleaned, report = clean_data(raw_dataset)
    assert report["initial_rows"] == 7043 or report["initial_rows"] == 7044
    assert report["blank_total_charges_found"] == 11
    assert report["zero_tenure_customers_preserved"] == 11
    assert report["final_missing_values"] == 0
    assert cleaned["TotalCharges"].dtype == "float64"
    assert cleaned["tenure"].dtype == "int32" or cleaned["tenure"].dtype == "int64"


# ------------------------------------------------------------------------------
# Reusable Analytical Function Tests
# ------------------------------------------------------------------------------

def test_get_churn_summary(synthetic_sample_df: pd.DataFrame):
    """Test get_churn_summary calculates correct counts and percentage."""
    cleaned, _ = clean_data(synthetic_sample_df)
    summary = get_churn_summary(cleaned)

    # In synthetic sample: 2 churned out of 5 -> 40.0%
    assert summary["total_customers"] == 5
    assert summary["churned_customers"] == 2
    assert summary["retained_customers"] == 3
    assert summary["churn_rate_pct"] == 40.0
    assert summary["retention_rate_pct"] == 60.0


def test_get_charge_and_tenure_metrics(cleaned_dataset: pd.DataFrame):
    """Test charge and tenure metrics return non-trivial realistic values."""
    metrics = get_charge_and_tenure_metrics(cleaned_dataset)
    overall = metrics["overall"]

    assert "avg_monthly_charges" in overall
    assert "avg_tenure" in overall
    assert 10 <= overall["avg_monthly_charges"] <= 120
    assert 0 <= overall["avg_tenure"] <= 72

    # Verify breakdown exists
    assert "Yes" in metrics["by_churn"]
    assert "No" in metrics["by_churn"]
    churned_avg_tenure = metrics["by_churn"]["Yes"]["avg_tenure"]
    retained_avg_tenure = metrics["by_churn"]["No"]["avg_tenure"]
    # In churn analysis, churned customers typically have lower average tenure than retained
    assert churned_avg_tenure < retained_avg_tenure


def test_get_churn_rate_by_category(cleaned_dataset: pd.DataFrame):
    """Test churn rate by Contract, InternetService, and PaymentMethod."""
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        cat_df = get_churn_rate_by_category(cleaned_dataset, category_col=col)
        assert not cat_df.empty
        assert col in cat_df.columns
        assert "Total Customers" in cat_df.columns
        assert "Churn Rate (%)" in cat_df.columns

        # Sum of total customers must equal total dataset rows
        assert cat_df["Total Customers"].sum() == len(cleaned_dataset)
        # Churn rate must be between 0 and 100
        assert (cat_df["Churn Rate (%)"] >= 0).all()
        assert (cat_df["Churn Rate (%)"] <= 100).all()


def test_get_tenure_distribution(cleaned_dataset: pd.DataFrame):
    """Test tenure distribution stats and cohort brackets."""
    dist = get_tenure_distribution(cleaned_dataset)
    stats = dist["stats"]
    brackets = dist["brackets"]

    assert "overall" in stats
    assert "churned" in stats
    assert "retained" in stats
    assert not brackets.empty
    assert "tenure_bracket" in brackets.columns
    assert brackets["Total Customers"].sum() == len(cleaned_dataset)


def test_filter_customers_and_cohort_metrics(cleaned_dataset: pd.DataFrame):
    """Test filtering and computing cohort financial impact."""
    m2m_fiber = filter_customers(
        cleaned_dataset,
        Contract="Month-to-month",
        InternetService="Fiber optic"
    )
    assert len(m2m_fiber) > 0
    assert (m2m_fiber["Contract"] == "Month-to-month").all()
    assert (m2m_fiber["InternetService"] == "Fiber optic").all()

    cohort_metrics = get_group_churn_metrics(m2m_fiber, group_name="M2M Fiber Optic")
    assert cohort_metrics["total_customers"] == len(m2m_fiber)
    assert cohort_metrics["churn_rate_pct"] > 0
    assert cohort_metrics["lost_monthly_revenue"] > 0


def test_compute_business_insights(cleaned_dataset: pd.DataFrame):
    """Test dynamic business insights generation with problem/evidence/recommendation."""
    insights = compute_business_insights(cleaned_dataset)
    assert len(insights) >= 3

    for insight in insights:
        assert "title" in insight
        assert "category" in insight
        assert "problem" in insight
        assert "evidence" in insight
        assert "recommendation" in insight
        assert insight["severity"] in ["High", "Medium", "Low"]
        # Ensure evidence contains actual numbers
        assert "%" in insight["evidence"]


def test_create_category_churn_chart(cleaned_dataset: pd.DataFrame):
    """Test Plotly bar chart generation for Contract and PaymentMethod."""
    fig_contract = create_category_churn_chart(cleaned_dataset, "Contract", "Contract Churn")
    assert fig_contract is not None
    assert len(fig_contract.data) > 0

    fig_payment = create_category_churn_chart(
        cleaned_dataset, "PaymentMethod", "Payment Method Churn", horizontal=True
    )
    assert fig_payment is not None
    assert len(fig_payment.data) > 0


def test_create_tenure_distribution_chart(cleaned_dataset: pd.DataFrame):
    """Test Plotly tenure distribution chart generation."""
    fig = create_tenure_distribution_chart(cleaned_dataset)
    assert fig is not None
    assert len(fig.data) == 2  # Retained and Churned histograms


def test_evaluate_kmeans_k_range(cleaned_dataset: pd.DataFrame):
    """Test k evaluation across candidate range 2-7."""
    eval_res = evaluate_kmeans_k_range(cleaned_dataset, k_min=2, k_max=7)
    assert eval_res["k_values"] == [2, 3, 4, 5, 6, 7]
    assert len(eval_res["inertias"]) == 6
    assert len(eval_res["silhouette_scores"]) == 6
    # Mathematical result on standard Telco data: k=4 has highest silhouette score
    assert eval_res["recommended_k"] == 4
    assert eval_res["best_silhouette_score"] > 0.4
    # Inertia must decrease strictly monotonically as k increases
    assert all(
        eval_res["inertias"][i] > eval_res["inertias"][i + 1]
        for i in range(len(eval_res["inertias"]) - 1)
    )


def test_fit_customer_clusters_reproducibility(cleaned_dataset: pd.DataFrame):
    """
    Critical requirement:
    Verify that fitting with the same random_state produces deterministic, reproducible cluster labels.
    """
    df_run1, meta1 = fit_customer_clusters(cleaned_dataset, n_clusters=4, random_state=42)
    df_run2, meta2 = fit_customer_clusters(cleaned_dataset, n_clusters=4, random_state=42)

    assert (df_run1["Cluster"] == df_run2["Cluster"]).all()
    assert meta1["inertia"] == pytest.approx(meta2["inertia"], rel=1e-6)
    assert meta1["silhouette_score"] == pytest.approx(meta2["silhouette_score"], rel=1e-6)


def test_fit_customer_clusters_assignment_and_summary(cleaned_dataset: pd.DataFrame):
    """Verify cluster labels are assigned to all records and summary table matches dataset."""
    clustered_df, meta = fit_customer_clusters(cleaned_dataset, n_clusters=4, random_state=42)

    # 1. Check cluster label assignment
    assert "Cluster" in clustered_df.columns
    assert set(clustered_df["Cluster"].unique()) == {0, 1, 2, 3}
    assert len(clustered_df) == len(cleaned_dataset)

    # 2. Check centroids
    centroids = meta["centroids"]
    assert len(centroids) == 4
    for c in centroids:
        assert "Cluster" in c
        assert 0 <= c["tenure"] <= 72
        assert 10 <= c["MonthlyCharges"] <= 125

    # 3. Check summary table integrity
    summary_table = meta["summary_table"]
    assert len(summary_table) == 4
    assert summary_table["Customer Count"].sum() == len(cleaned_dataset)
    assert (summary_table["Observed Churn Rate (%)"] >= 0).all()
    assert (summary_table["Observed Churn Rate (%)"] <= 100).all()
    assert "Segment Archetype" in summary_table.columns
    assert "Strategic Focus" in summary_table.columns


def test_kmeans_visualizations(cleaned_dataset: pd.DataFrame):
    """Test Plotly diagnostic and scatter visualization generation."""
    eval_res = evaluate_kmeans_k_range(cleaned_dataset)
    fig_diag = create_elbow_and_silhouette_chart(eval_res, selected_k=4)
    assert fig_diag is not None
    assert len(fig_diag.data) >= 3

    clustered_df, meta = fit_customer_clusters(cleaned_dataset, n_clusters=4)
    fig_scatter = create_cluster_scatter_plot(clustered_df, centroids=meta["centroids"])
    assert fig_scatter is not None
    # Must contain traces for each cluster + centroids
    assert len(fig_scatter.data) >= 4


def test_train_decision_tree_model(cleaned_dataset: pd.DataFrame):
    """Test Decision Tree training workflow, CV selection, and held-out test evaluation."""
    results = train_decision_tree_model(cleaned_dataset, test_size=0.20, random_state=42)

    # 1. Split integrity
    split = results["split_summary"]
    assert split["train_size"] + split["test_size"] == len(cleaned_dataset)
    assert split["test_size"] == 1409
    # Stratification check: churn proportions in train and test should match
    assert abs(split["train_churn_rate_pct"] - split["test_churn_rate_pct"]) < 0.5

    # 2. Model selection on train data
    assert results["selected_depth"] in results["candidate_depths"]
    best_cv_f1 = results["cv_results"]["Mean CV F1"].max()
    expected_depth = results["cv_results"].loc[results["cv_results"]["Mean CV F1"] == best_cv_f1, "Depth"].iloc[0]
    assert results["selected_depth"] == expected_depth
    assert len(results["cv_results"]) == len(results["candidate_depths"])

    # 3. Held-out test evaluation metrics
    metrics = results["test_metrics"]
    assert 0.70 <= metrics["accuracy"] <= 0.85
    assert 0.50 <= metrics["precision"] <= 0.75
    assert 0.50 <= metrics["recall"] <= 0.75
    assert 0.50 <= metrics["f1"] <= 0.75

    # 4. Confusion matrix integrity
    cm = results["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == split["test_size"]

    # 5. Baseline comparison
    baseline = results["baseline_metrics"]
    assert baseline["majority_class"] == 0
    assert 0.70 <= baseline["baseline_accuracy"] <= 0.75
    # Model should exhibit positive accuracy lift over naive guessing
    assert baseline["accuracy_lift_pct"] > 0

    # 6. Feature importances
    fi_df = results["feature_importance"]
    assert not fi_df.empty
    assert "feature" in fi_df.columns
    assert "importance" in fi_df.columns
    assert abs(fi_df["importance"].sum() - 1.0) < 0.05


def test_decision_tree_pipeline_unseen_categories(cleaned_dataset: pd.DataFrame):
    """
    Critical requirement:
    Verify that OneHotEncoder(handle_unknown='ignore') safely handles unseen category values
    without crashing at inference time.
    """
    results = train_decision_tree_model(cleaned_dataset)
    pipeline = results["pipeline"]

    # Input with unseen categories: e.g. Contract='3-year', InternetService='Satellite'
    unseen_customer = pd.DataFrame([{
        "Contract": "3-year",
        "tenure": 10,
        "MonthlyCharges": 70.0,
        "InternetService": "Satellite",
        "PaymentMethod": "Cryptocurrency",
        "TechSupport": "Unknown",
        "PaperlessBilling": "Maybe",
        "SeniorCitizen": 0,
    }])

    # Must predict without error
    pred = pipeline.predict(unseen_customer)
    prob = pipeline.predict_proba(unseen_customer)
    assert pred[0] in [0, 1]
    assert len(prob[0]) == 2


def test_predict_customer_churn_risk(cleaned_dataset: pd.DataFrame):
    """Test individual customer churn prediction and risk score logic."""
    results = train_decision_tree_model(cleaned_dataset)
    pipeline = results["pipeline"]

    # Typical high-risk profile: Month-to-month, short tenure, Fiber optic, Electronic check
    high_risk_input = {
        "Contract": "Month-to-month",
        "tenure": 1,
        "MonthlyCharges": 95.0,
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "PaperlessBilling": "Yes",
        "SeniorCitizen": 0,
    }
    outcome = predict_customer_churn_risk(pipeline, high_risk_input)

    assert outcome["predicted_class"] == 1
    assert outcome["predicted_label"] == "High Churn Risk (Class 1)"
    assert outcome["uncalibrated_risk_score"] > 0.5
    assert "disclaimer" in outcome
    assert "leaf node" in outcome["disclaimer"]

    # Boundary test: negative tenure must raise ValueError
    invalid_input = dict(high_risk_input)
    invalid_input["tenure"] = -5
    with pytest.raises(ValueError):
        predict_customer_churn_risk(pipeline, invalid_input)


def test_decision_tree_visualizations(cleaned_dataset: pd.DataFrame):
    """Test Plotly diagnostic charts for Decision Tree."""
    results = train_decision_tree_model(cleaned_dataset)

    fig_cm = create_confusion_matrix_chart(results["confusion_matrix"])
    assert fig_cm is not None
    assert len(fig_cm.data) > 0

    fig_fi = create_feature_importance_chart(results["feature_importance"])
    assert fig_fi is not None
    assert len(fig_fi.data) > 0

    fig_cv = create_depth_cv_chart(results["cv_results"], selected_depth=5)
    assert fig_cv is not None
    assert len(fig_cv.data) >= 1


# ------------------------------------------------------------------------------
# Business Insights: 4-Part Structure Tests (Overview, Segments, Model)
# ------------------------------------------------------------------------------

def test_compute_business_insights_limitation_key(cleaned_dataset: pd.DataFrame):
    """Verify the 4-part insight structure includes the 'limitation' key."""
    insights = compute_business_insights(cleaned_dataset)
    assert len(insights) >= 3

    required_keys = {"id", "title", "category", "severity", "problem", "evidence", "recommendation", "limitation"}
    for insight in insights:
        missing = required_keys - insight.keys()
        assert not missing, f"Insight missing keys: {missing} in insight '{insight.get('title')}'"
        # All four narrative fields must be non-empty strings
        for key in ("problem", "evidence", "recommendation", "limitation"):
            assert isinstance(insight[key], str) and len(insight[key]) > 0, (
                f"Expected non-empty string for '{key}' in '{insight.get('title')}'"
            )
        # Evidence must contain at least one percentage from real computed data
        assert "%" in insight["evidence"], (
            f"Evidence in '{insight.get('title')}' must contain a percentage value"
        )


def test_compute_business_insights_dynamic_sensitivity(cleaned_dataset: pd.DataFrame):
    """Verify insight text updates dynamically when the underlying data slice changes."""
    full_insights = compute_business_insights(cleaned_dataset)
    full_evidence = {i["id"]: i["evidence"] for i in full_insights}

    # Restrict to first-year customers only
    early_df = cleaned_dataset[cleaned_dataset["tenure"] <= 12].copy()
    early_insights = compute_business_insights(early_df)
    early_evidence = {i["id"]: i["evidence"] for i in early_insights}

    # The contract risk insight must appear in both (both have >1 contract type)
    # and the count/percentage in evidence must differ across the two cohorts
    if "contract_risk" in full_evidence and "contract_risk" in early_evidence:
        assert full_evidence["contract_risk"] != early_evidence["contract_risk"], (
            "Expected contract_risk evidence to differ between full and early-tenure cohorts"
        )


def test_generate_segment_insights(cleaned_dataset: pd.DataFrame):
    """Test 4-part structure of generate_segment_insights across all keys."""
    _, cluster_meta = fit_customer_clusters(cleaned_dataset, n_clusters=4, random_state=42)
    insights = generate_segment_insights(cluster_meta)

    # Must return at least 2 insights (highest + lowest churn cluster)
    assert len(insights) >= 2

    required_keys = {"id", "title", "category", "severity", "problem", "evidence", "recommendation", "limitation"}
    for insight in insights:
        missing = required_keys - insight.keys()
        assert not missing, f"Segment insight missing keys: {missing} in '{insight.get('title')}'"
        for key in ("problem", "evidence", "recommendation", "limitation"):
            assert isinstance(insight[key], str) and len(insight[key]) > 0
        # Evidence must cite a percentage churn rate
        assert "%" in insight["evidence"]
        # Severity must be a recognized value
        assert insight["severity"] in ("High", "Medium", "Low")

    # IDs: highest and lowest churn insights must always be present
    ids = [i["id"] for i in insights]
    assert "highest_churn_segment" in ids
    assert "stable_retention_segment" in ids


def test_generate_segment_insights_dynamic_sensitivity(cleaned_dataset: pd.DataFrame):
    """Verify segment insight evidence differs when k changes (different cluster boundaries)."""
    _, meta_k3 = fit_customer_clusters(cleaned_dataset, n_clusters=3, random_state=42)
    _, meta_k5 = fit_customer_clusters(cleaned_dataset, n_clusters=5, random_state=42)

    insights_k3 = generate_segment_insights(meta_k3)
    insights_k5 = generate_segment_insights(meta_k5)

    evidence_k3 = {i["id"]: i["evidence"] for i in insights_k3}
    evidence_k5 = {i["id"]: i["evidence"] for i in insights_k5}

    # Different k produces different cluster stats, so highest-churn evidence must differ
    assert evidence_k3.get("highest_churn_segment") != evidence_k5.get("highest_churn_segment"), (
        "Expected highest_churn_segment evidence to differ between k=3 and k=5"
    )


def test_generate_model_insights(cleaned_dataset: pd.DataFrame):
    """Test 4-part structure of generate_model_insights across all keys."""
    results = train_decision_tree_model(cleaned_dataset)
    insights = generate_model_insights(results)

    # Must return exactly 3 insights
    assert len(insights) == 3

    required_keys = {"id", "title", "category", "severity", "problem", "evidence", "recommendation", "limitation"}
    for insight in insights:
        missing = required_keys - insight.keys()
        assert not missing, f"Model insight missing keys: {missing} in '{insight.get('title')}'"
        for key in ("problem", "evidence", "recommendation", "limitation"):
            assert isinstance(insight[key], str) and len(insight[key]) > 0
        # Evidence must contain at least one numeric figure
        assert any(char.isdigit() for char in insight["evidence"]), (
            f"Evidence in '{insight.get('title')}' must contain numeric values"
        )

    # All three required insight IDs must be present
    ids = [i["id"] for i in insights]
    assert "primary_feature_reliance" in ids
    assert "classification_tradeoff" in ids
    assert "class_imbalance_paradox" in ids

    # Feature reliance: evidence must cite a Gini percentage
    fi_insight = next(i for i in insights if i["id"] == "primary_feature_reliance")
    assert "%" in fi_insight["evidence"]

    # Limitation must NOT assert bare causality (must be qualified)
    fi_limitation = fi_insight["limitation"].lower()
    assert "not" in fi_limitation, (
        "primary_feature_reliance limitation must contain a negation (e.g., 'not causal')"
    )


def test_generate_model_insights_dynamic_values(cleaned_dataset: pd.DataFrame):
    """Verify model insight evidence reflects real computed model metrics (not hardcoded)."""
    results = train_decision_tree_model(cleaned_dataset)
    insights = generate_model_insights(results)

    cm = results["confusion_matrix"]
    tp, fn = cm["tp"], cm["fn"]

    # Classification trade-off evidence must contain real TP or FN counts from confusion matrix
    tradeoff = next(i for i in insights if i["id"] == "classification_tradeoff")
    evidence_text = tradeoff["evidence"]
    assert str(tp) in evidence_text or str(fn) in evidence_text, (
        f"Expected TP={tp} or FN={fn} from confusion matrix to appear in classification_tradeoff evidence"
    )

    # Class imbalance paradox must cite the real baseline accuracy
    base_acc = results["baseline_metrics"]["baseline_accuracy_pct"]
    paradox = next(i for i in insights if i["id"] == "class_imbalance_paradox")
    assert f"{base_acc:.1f}" in paradox["evidence"], (
        f"Expected baseline accuracy {base_acc:.1f}% to appear in class_imbalance_paradox evidence"
    )


def test_build_insight_card_html_structure_and_safety():
    """
    Validates that build_insight_card_html:
    1. Generates unindented HTML without internal blank lines (prevents markdown-it code-block bug).
    2. Correctly contains all 4 structured section labels.
    3. Safely escapes dangerous HTML characters to prevent XSS/malformed tags.
    """
    raw_insight = {
        "title": "Contract Risk & Evaluation",
        "category": "Contract Structure",
        "severity": "High",
        "problem": "Customers on <month-to-month> contracts leave at high rates.",
        "evidence": "42.7% churn (1,655 < 3,875 & rate > 20%).",
        "recommendation": "Introduce 'annual' discounts & incentives.",
        "limitation": "Self-selection bias; correlation != causation.",
    }

    card_html = build_insight_card_html(raw_insight)

    # 1. Output must be a non-empty string
    assert isinstance(card_html, str)
    assert len(card_html) > 100

    # 2. Must contain required CSS classes and structural elements
    assert 'class="insight-card"' in card_html
    assert 'class="insight-badge"' in card_html
    assert 'class="insight-title"' in card_html
    assert 'class="insight-section-label"' in card_html
    assert 'class="insight-text"' in card_html
    assert 'class="insight-limitation-box"' in card_html

    # 3. Must contain all 4 section headings
    assert "🔴 1. Problem Identified" in card_html
    assert "📊 2. Measured Evidence" in card_html
    assert "💡 3. Recommended Investigation" in card_html
    assert "⚠️ 4. Analytical Limitation" in card_html

    # 4. Must properly escape dangerous HTML entities (<, >, &)
    assert "&lt;month-to-month&gt;" in card_html
    assert "&amp;" in card_html
    assert "<month-to-month>" not in card_html

    # 5. Must NOT trigger Markdown indented code blocks when parsed by markdown-it
    try:
        import markdown_it
        md = markdown_it.MarkdownIt()
        rendered = md.render(card_html)
        assert "<pre>" not in rendered, "HTML card must not trigger indented code block in Markdown parser"
        assert "<code>" not in rendered, "HTML card must not trigger code block in Markdown parser"
    except ImportError:
        pass


def test_build_insight_card_html_on_actual_dataset(cleaned_dataset: pd.DataFrame):
    """Verifies that all overview insights generated from actual data produce valid HTML cards."""
    insights = compute_business_insights(cleaned_dataset)
    assert len(insights) > 0
    for insight in insights:
        html_card = build_insight_card_html(insight)
        assert 'class="insight-card"' in html_card
        assert "🔴 1. Problem Identified" in html_card
        assert "📊 2. Measured Evidence" in html_card
        assert "💡 3. Recommended Investigation" in html_card
        assert "⚠️ 4. Analytical Limitation" in html_card
        assert insight["problem"] in html_card or html_card.count("insight-text") >= 3
