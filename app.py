"""
ChurnLens: Customer Churn Analytics and Risk Predictor
Main Streamlit Application

Architecture:
- Data Layer: churnlens.py (loading, validation, safe cleaning, statistical aggregations)
- Visualization Layer: visualizations.py (Plotly figures with unified styling)
- UI / Presentation Layer: app.py (Streamlit dashboard with tabs, KPI cards, interactive filters, insights)
"""

import html
from typing import Tuple
import pandas as pd
import streamlit as st

from churnlens import (
    clean_data,
    compute_business_insights,
    evaluate_kmeans_k_range,
    fit_customer_clusters,
    generate_model_insights,
    generate_segment_insights,
    get_charge_and_tenure_metrics,
    get_churn_summary,
    load_data,
    predict_customer_churn_risk,
    train_decision_tree_model,
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

# Set page configuration
st.set_page_config(
    page_title="ChurnLens | Customer Churn Analytics & Risk Predictor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics, metric cards, and clean typography
st.markdown(
    """
    <style>
    /* Global Typography & Background adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header hero styling */
    .hero-container {
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        border-left: 5px solid #3b82f6;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 0;
    }

    /* KPI Cards Styling */
    .kpi-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.12);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-value-danger {
        color: #ef4444;
    }
    .kpi-value-primary {
        color: #2563eb;
    }
    .kpi-subtext {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.4rem;
    }

    /* Business Insight Card Styling */
    .insight-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        margin-bottom: 1.2rem;
        border-left: 5px solid #3b82f6;
    }
    .insight-card-high {
        border-left-color: #ef4444;
    }
    .insight-card-medium {
        border-left-color: #f59e0b;
    }
    .insight-card-low {
        border-left-color: #10b981;
    }
    .insight-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
    }
    .badge-category {
        background-color: #f1f5f9;
        color: #475569;
        margin-left: 0.4rem;
    }
    .insight-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.6rem;
    }
    .insight-section-label {
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.65rem;
    }
    .insight-text {
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.5;
        margin-bottom: 0.4rem;
    }
    .insight-limitation-box {
        font-size: 0.84rem;
        color: #475569;
        line-height: 1.45;
        background-color: #f8fafc;
        border-left: 3px solid #94a3b8;
        padding: 0.55rem 0.75rem;
        border-radius: 4px;
        margin-top: 0.35rem;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading and verifying dataset...")
def get_prepared_data() -> Tuple[pd.DataFrame, dict]:
    """
    Loads and cleans the dataset once, caching the result in memory.
    """
    raw_df = load_data()
    cleaned_df, report = clean_data(raw_df)
    return cleaned_df, report


@st.cache_data(show_spinner="Evaluating candidate k values (2-7)...")
def get_kmeans_evaluation(df_data: pd.DataFrame) -> dict:
    """
    Runs K-Means evaluation across k=2 to 7, returning inertia and silhouette scores.
    """
    return evaluate_kmeans_k_range(df_data, features=["tenure", "MonthlyCharges"])


@st.cache_data(show_spinner="Fitting customer clusters...")
def get_fitted_clusters(df_data: pd.DataFrame, n_clusters: int) -> Tuple[pd.DataFrame, dict]:
    """
    Fits final K-Means model with fixed seed, returning clustered data and summary metadata.
    """
    return fit_customer_clusters(df_data, features=["tenure", "MonthlyCharges"], n_clusters=n_clusters)


@st.cache_resource(show_spinner="Training and cross-validating Decision Tree classifier...")
def get_trained_tree(df_data: pd.DataFrame) -> dict:
    """
    Trains Decision Tree on df_data with 80/20 stratified split and CV model selection.
    """
    return train_decision_tree_model(df_data)


def render_insight_card(insight: dict) -> None:
    """
    Renders a unified, data-backed 4-part Business Insight Card:
    1. Problem: Pattern or potential friction point identified.
    2. Evidence: Measured values, percentages, and customer counts.
    3. Recommendation: Reasonable tactical action to investigate next.
    4. Limitation: Explicit boundaries and what cannot be causally claimed.

    Uses st.html() to render raw trusted HTML directly without Markdown parsing interference,
    falling back to st.markdown(card_html, unsafe_allow_html=True) if st.html is unavailable.
    """
    card_html = build_insight_card_html(insight)
    if hasattr(st, "html"):
        st.html(card_html)
    else:
        st.markdown(card_html, unsafe_allow_html=True)


# Main Application Header
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🔍 ChurnLens: Customer Churn Analytics & Risk Predictor</div>
        <p class="hero-subtitle">
            Exploratory intelligence, customer segmentation, and churn risk classification based on the IBM Telco dataset.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load dataset with error handling
try:
    df, clean_report = get_prepared_data()
except Exception as e:
    st.error(
        f"⚠️ **Error loading dataset**: {e}\n\n"
        "Please ensure `WA_Fn-UseC_-Telco-Customer-Churn.csv` is placed inside the `data/` directory."
    )
    st.stop()


# ==============================================================================
# SIDEBAR FILTERS
# ==============================================================================
st.sidebar.header("🎯 Cohort Filters")
st.sidebar.caption("Filter data dynamically across all dashboard metrics and charts.")

# Contract filter
available_contracts = sorted(df["Contract"].unique())
selected_contracts = st.sidebar.multiselect(
    "Contract Type",
    options=available_contracts,
    default=available_contracts,
    help="Filter by customer contract commitment duration.",
)

# Internet Service filter
available_internet = sorted(df["InternetService"].unique())
selected_internet = st.sidebar.multiselect(
    "Internet Service",
    options=available_internet,
    default=available_internet,
    help="Filter by primary internet connection technology.",
)

# Payment Method filter
available_pm = sorted(df["PaymentMethod"].unique())
selected_pm = st.sidebar.multiselect(
    "Payment Method",
    options=available_pm,
    default=available_pm,
    help="Filter by customer billing payment channel.",
)

# Senior Citizen filter
senior_choice = st.sidebar.radio(
    "Senior Citizen Cohort",
    options=["All Customers", "Senior Citizens (65+)", "Non-Seniors (<65)"],
    index=0,
)

# Apply filters consistently
filtered_df = df.copy()

if selected_contracts:
    filtered_df = filtered_df[filtered_df["Contract"].isin(selected_contracts)]
else:
    filtered_df = filtered_df.iloc[0:0]

if selected_internet:
    filtered_df = filtered_df[filtered_df["InternetService"].isin(selected_internet)]
else:
    filtered_df = filtered_df.iloc[0:0]

if selected_pm:
    filtered_df = filtered_df[filtered_df["PaymentMethod"].isin(selected_pm)]
else:
    filtered_df = filtered_df.iloc[0:0]

if senior_choice == "Senior Citizens (65+)":
    filtered_df = filtered_df[filtered_df["SeniorCitizen"] == 1]
elif senior_choice == "Non-Seniors (<65)":
    filtered_df = filtered_df[filtered_df["SeniorCitizen"] == 0]

# Sidebar Cohort Summary
total_count = len(df)
filtered_count = len(filtered_df)
pct_of_total = (filtered_count / total_count * 100) if total_count > 0 else 0.0

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Filtered Sample**: {filtered_count:,} / {total_count:,} ({pct_of_total:.1f}%)")

if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Data Quality Verification:")
st.sidebar.caption(
    f"• Rows Analyzed: {clean_report['final_rows']:,}\n"
    f"• Zero-tenure Preserved: {clean_report['zero_tenure_customers_preserved']}\n"
    f"• Duplicates Removed: {clean_report['duplicates_removed']}"
)


# ==============================================================================
# THREE TAB STRUCTURE
# ==============================================================================
tab_overview, tab_segments, tab_churn_risk = st.tabs(
    ["📊 Overview & Insights", "👥 Customer Segments", "🎯 Churn Risk"]
)


# ------------------------------------------------------------------------------
# TAB 1: OVERVIEW & INSIGHTS
# ------------------------------------------------------------------------------
with tab_overview:
    if filtered_df.empty:
        st.warning("⚠️ No customer records match the selected sidebar filters. Please adjust your selections.")
    else:
        # 1. Compute dynamic metrics
        summary = get_churn_summary(filtered_df)
        charge_metrics = get_charge_and_tenure_metrics(filtered_df)["overall"]

        # 2. KPI Cards Row
        st.subheader("Key Performance Indicators", divider="blue")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Total Customers</div>
                    <div class="kpi-value kpi-value-primary">{summary['total_customers']:,}</div>
                    <div class="kpi-subtext">Active cohort sample size</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Cohort Churn Rate</div>
                    <div class="kpi-value kpi-value-danger">{summary['churn_rate_pct']:.1f}%</div>
                    <div class="kpi-subtext">{summary['churned_customers']:,} churned vs {summary['retained_customers']:,} retained</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Avg Monthly Charges</div>
                    <div class="kpi-value">${charge_metrics['avg_monthly_charges']:.2f}</div>
                    <div class="kpi-subtext">Median: ${charge_metrics['median_monthly_charges']:.2f} / month</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with kpi4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Average Tenure</div>
                    <div class="kpi-value">{charge_metrics['avg_tenure']:.1f} <span style="font-size: 1rem; font-weight: normal; color: #64748b;">mo</span></div>
                    <div class="kpi-subtext">Median: {charge_metrics['median_tenure']:.0f} months</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Interactive Visualizations (2x2 Grid)
        st.subheader("Exploratory Visualizations", divider="blue")
        st.caption("Interactive charts comparing customer attrition across service tiers and account attributes.")

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            fig_contract = create_category_churn_chart(
                filtered_df,
                category_col="Contract",
                title="Churn Rate by Contract Type",
            )
            st.plotly_chart(fig_contract, use_container_width=True)

        with row1_col2:
            fig_internet = create_category_churn_chart(
                filtered_df,
                category_col="InternetService",
                title="Churn Rate by Internet Service",
            )
            st.plotly_chart(fig_internet, use_container_width=True)

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            fig_payment = create_category_churn_chart(
                filtered_df,
                category_col="PaymentMethod",
                title="Churn Rate by Payment Method",
                horizontal=True,
            )
            st.plotly_chart(fig_payment, use_container_width=True)

        with row2_col2:
            fig_tenure = create_tenure_distribution_chart(filtered_df)
            st.plotly_chart(fig_tenure, use_container_width=True)

        # 4. Business Insights Section
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Data-Backed Business Insights", divider="blue")
        st.caption(
            "Synthesized problem statements, empirical evidence from the loaded data, "
            "and tactical business initiatives to investigate."
        )

        insights = compute_business_insights(filtered_df)

        if not insights:
            st.info("No specific categorical friction patterns identified for this filtered sample.")
        else:
            ins_col1, ins_col2 = st.columns(2)
            for idx, insight in enumerate(insights):
                target_col = ins_col1 if idx % 2 == 0 else ins_col2
                with target_col:
                    render_insight_card(insight)

        st.info(
            "📌 **Analytical Integrity Note for Interviews**: "
            "The observed correlations above (e.g., higher churn among Month-to-month and Electronic Check users) "
            "demonstrate empirical associations in historical data. They do not prove that contract length or payment channel "
            "are the sole causal drivers of churn. Recommended actions should be tested via controlled experiments (A/B testing) "
            "or targeted pilot retention initiatives."
        )


# ------------------------------------------------------------------------------
# TAB 2: CUSTOMER SEGMENTS (K-MEANS CLUSTERING)
# ------------------------------------------------------------------------------
with tab_segments:
    if len(filtered_df) < 10:
        st.warning("⚠️ Insufficient customer records matching filters to perform clustering. Please broaden your selection.")
    else:
        st.subheader("Customer Segmentation (K-Means Clustering)", divider="blue")
        st.markdown(
            """
            This module segments customers based on two continuous behavioral dimensions:
            - **`tenure`**: Customer relationship duration (months)
            - **`MonthlyCharges`**: Financial commitment level ($ USD)
            
            Before clustering, both features are standardized using **`StandardScaler`** (mean = 0, standard deviation = 1) 
            to prevent the wider range of monthly charges ($18–$120) from artificially dominating tenure (0–72 months) 
            in Euclidean distance calculations.
            """
        )

        # 1. Evaluate k from 2 through 7
        eval_results = get_kmeans_evaluation(filtered_df)

        st.markdown("#### 1. Optimal Cluster Count Diagnostics (k = 2 to 7)")
        
        diag_col1, diag_col2 = st.columns([1, 2])
        with diag_col1:
            selected_k = st.slider(
                "Select Number of Clusters (k):",
                min_value=2,
                max_value=7,
                value=int(eval_results["recommended_k"]),
                step=1,
                help="Adjust k to inspect how customer boundaries and silhouette cohesion change.",
            )

            rec_k = eval_results["recommended_k"]
            best_sil = eval_results["best_silhouette_score"]
            st.markdown(
                f"""
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem; font-size: 0.85rem; color: #334155;">
                    <div style="font-weight: 600; color: #0f172a; margin-bottom: 0.3rem;">📊 Mathematical Justification</div>
                    <div>• <b>Peak Silhouette Score</b>: <b>{best_sil:.4f}</b> at <b>k = {rec_k}</b></div>
                    <div style="margin-top: 0.3rem;">• <b>Elbow Inflection</b>: Inertia drops steeply up to k={rec_k}, after which marginal variance reduction tapers off.</div>
                    <div style="margin-top: 0.4rem; color: #64748b; font-size: 0.8rem;">
                        <i>Note: Both mathematical metrics support k={rec_k} as the most cohesive separation without over-segmenting.</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with diag_col2:
            fig_diag = create_elbow_and_silhouette_chart(eval_results, selected_k=selected_k)
            st.plotly_chart(fig_diag, use_container_width=True)

        st.markdown("---")

        # 2. Fit Final Model for Selected k
        clustered_df, cluster_meta = get_fitted_clusters(filtered_df, n_clusters=selected_k)
        summary_table = cluster_meta["summary_table"]

        st.markdown(f"#### 2. Customer Segment Distribution (k = {selected_k})")
        st.caption(
            "Scatter plot showing customer positioning along Tenure vs. Monthly Charges. "
            "Diamonds represent cluster centroids (average values in original units)."
        )

        fig_scatter = create_cluster_scatter_plot(
            clustered_df,
            centroids=cluster_meta["centroids"],
            x_col="tenure",
            y_col="MonthlyCharges",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 3. Cluster Summary Table
        st.markdown("#### 3. Empirical Cluster Profiles")
        st.caption(
            "All statistics below are calculated directly from the segmented records. "
            "Archetype descriptions are assigned after evaluating empirical cluster centroids."
        )

        display_cols = [
            "Cluster",
            "Segment Archetype",
            "Customer Count",
            "Share of Base (%)",
            "Avg Tenure (mos)",
            "Avg Monthly Charges ($)",
            "Observed Churn Rate (%)",
        ]
        st.dataframe(
            summary_table[display_cols].style.format({
                "Customer Count": "{:,}",
                "Share of Base (%)": "{:.1f}%",
                "Avg Tenure (mos)": "{:.1f}",
                "Avg Monthly Charges ($)": "${:.2f}",
                "Observed Churn Rate (%)": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # 4. Deep-Dive Segment Persona Cards
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 4. Segment Characteristics & Strategic Retention Actions")

        card_cols = st.columns(2)
        for idx, row in summary_table.iterrows():
            target_col = card_cols[idx % 2]
            with target_col:
                churn_rate = row["Observed Churn Rate (%)"]
                badge_bg = "#fee2e2" if churn_rate > 35 else "#fef3c7" if churn_rate > 15 else "#dcfce7"
                badge_fg = "#991b1b" if churn_rate > 35 else "#92400e" if churn_rate > 15 else "#166534"
                risk_label = "High Risk" if churn_rate > 35 else "Moderate Risk" if churn_rate > 15 else "Low Risk"

                archetype_escaped = html.escape(str(row["Segment Archetype"]))
                focus_escaped = html.escape(str(row["Strategic Focus"]))
                persona_card_html = (
                    f'<div class="insight-card" style="border-left-color: {badge_fg};">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">'
                    f'<span style="font-weight: 700; font-size: 1.05rem; color: #0f172a;">{row["Cluster"]}: {archetype_escaped}</span>'
                    f'<span style="background-color: {badge_bg}; color: {badge_fg}; padding: 0.15rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600;">{risk_label} ({churn_rate:.1f}%)</span>'
                    f'</div>'
                    f'<div style="font-size: 0.84rem; color: #64748b; margin-bottom: 0.6rem;">'
                    f'<b>{row["Customer Count"]:,} customers</b> ({row["Share of Base (%)"]:.1f}% of cohort) | '
                    f'Avg Tenure: <b>{row["Avg Tenure (mos)"]:.1f} mos</b> | '
                    f'Avg Monthly Bill: <b>${row["Avg Monthly Charges ($)"]:.2f}</b>'
                    f'</div>'
                    f'<div class="insight-section-label">🎯 Strategic Focus & Recommended Action</div>'
                    f'<div class="insight-text">{focus_escaped}</div>'
                    f'</div>'
                )
                if hasattr(st, "html"):
                    st.html(persona_card_html)
                else:
                    st.markdown(persona_card_html, unsafe_allow_html=True)

        # Mathematical Integrity Callout
        st.info(
            "📌 **Analytical Integrity Note for Interviews**: "
            "Clusters describe mathematical proximity in the standardized feature space based on Euclidean distance minimization. "
            "They do not represent ground-truth or objectively 'correct' customer types. In business practice, customer segmentation "
            "serves as a heuristic tool to personalize communications and prioritize retention resources."
        )

        # 5. Data-Backed Business Insights from Clustering
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Data-Backed Business Insights (Clustering)", divider="blue")
        st.caption(
            "Each insight below is derived exclusively from the computed cluster characteristics above — "
            "no hardcoded values. Analytical limitations are explicitly stated."
        )

        seg_insights = generate_segment_insights(cluster_meta)
        if not seg_insights:
            st.info("Insufficient cluster variation to generate comparative insights.")
        else:
            seg_col1, seg_col2 = st.columns(2)
            for idx, insight in enumerate(seg_insights):
                target_col = seg_col1 if idx % 2 == 0 else seg_col2
                with target_col:
                    render_insight_card(insight)


# ------------------------------------------------------------------------------
# TAB 3: CHURN RISK (DECISION TREE CLASSIFIER)
# ------------------------------------------------------------------------------
with tab_churn_risk:
    st.subheader("Churn Risk Predictor (Decision Tree Classifier)", divider="blue")
    st.markdown(
        """
        Supervised machine learning model trained to classify customer attrition risk using 
        **8 interpretable customer features**:
        `Contract`, `tenure`, `MonthlyCharges`, `InternetService`, `PaymentMethod`, `TechSupport`, 
        `PaperlessBilling`, and `SeniorCitizen`.
        
        *Exclusions*: `customerID` (identifier) and `TotalCharges` (excluded to avoid collinearity with tenure × charges).
        """
    )

    # 1. Train & evaluate the model pipeline
    model_results = get_trained_tree(df)
    test_metrics = model_results["test_metrics"]
    baseline_metrics = model_results["baseline_metrics"]
    split_info = model_results["split_summary"]

    st.markdown("#### 1. Held-Out Test Evaluation & Baseline Comparison")
    st.caption(
        f"Evaluated on a held-out test set of {split_info['test_size']:,} customers (20% stratified sample). "
        f"Selected optimal max_depth = {model_results['selected_depth']} using 5-fold cross-validation on the training set."
    )

    # Performance Metrics Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Test Accuracy</div>
                <div class="kpi-value kpi-value-primary">{test_metrics['accuracy_pct']:.1f}%</div>
                <div class="kpi-subtext">Overall correct predictions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Majority Baseline</div>
                <div class="kpi-value" style="color: #64748b;">{baseline_metrics['baseline_accuracy_pct']:.1f}%</div>
                <div class="kpi-subtext">Naive 'No Churn' guess</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Churn Precision</div>
                <div class="kpi-value kpi-value-danger">{test_metrics['precision_pct']:.1f}%</div>
                <div class="kpi-subtext">True churners / predicted churners</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Churn Recall</div>
                <div class="kpi-value kpi-value-danger">{test_metrics['recall_pct']:.1f}%</div>
                <div class="kpi-subtext">Detected churners / all churners</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Churn F1 Score</div>
                <div class="kpi-value kpi-value-primary">{test_metrics['f1_pct']:.1f}%</div>
                <div class="kpi-subtext">Harmonic mean of Prec. & Rec.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Explanatory Callout on Class Imbalance
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"⚖️ **Why Accuracy Alone is Misleading (The Class Imbalance Paradox)**: "
        f"In this dataset, **{baseline_metrics['baseline_accuracy_pct']:.1f}%** of customers do not churn. "
        f"A naive baseline model predicting *'No Churn'* for every single customer achieves **{baseline_metrics['baseline_accuracy_pct']:.1f}% accuracy**, "
        f"yet it detects **0% of churners** (0 recall, 0 F1). In contrast, our Decision Tree achieves **{test_metrics['recall_pct']:.1f}% recall**, "
        f"correctly identifying **{model_results['confusion_matrix']['tp']:,} churners** out of the {model_results['confusion_matrix']['tp'] + model_results['confusion_matrix']['fn']:,} actual churners in the test set."
    )

    # Diagnostic Visualizations: Confusion Matrix & Feature Importance
    st.markdown("#### 2. Model Diagnostics: Error Distribution & Feature Reliance")
    tree_row1, tree_row2 = st.columns(2)

    with tree_row1:
        fig_cm = create_confusion_matrix_chart(model_results["confusion_matrix"])
        st.plotly_chart(fig_cm, use_container_width=True)

    with tree_row2:
        fig_fi = create_feature_importance_chart(model_results["feature_importance"], top_n=8)
        st.plotly_chart(fig_fi, use_container_width=True)

    st.caption(
        "📌 **Feature Importance Notice**: Gini importance quantifies the total reduction in impurity brought by that feature. "
        "It indicates mathematical **model reliance** when deciding splits, **not causality**."
    )

    # Cross-Validation Expander
    with st.expander("🔍 View Training Cross-Validation Tuning (max_depth 2 to 7)"):
        cv_col1, cv_col2 = st.columns([1, 2])
        with cv_col1:
            st.dataframe(
                model_results["cv_results"].style.format({
                    "Mean CV F1": "{:.4f}",
                    "Std CV F1": "{:.4f}",
                    "Mean CV Accuracy": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        with cv_col2:
            fig_cv = create_depth_cv_chart(model_results["cv_results"], selected_depth=model_results["selected_depth"])
            st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("---")

    # 3. Interactive Customer Risk Prediction Form
    st.markdown("#### 3. Individual Customer Churn Risk Simulator")
    st.caption(
        "Enter customer attributes below to evaluate churn risk using the exact preprocessing and decision tree pipeline."
    )

    with st.form("customer_risk_form"):
        form_r1_c1, form_r1_c2, form_r1_c3 = st.columns(3)
        with form_r1_c1:
            input_contract = st.selectbox(
                "Contract Type",
                options=["Month-to-month", "One year", "Two year"],
                index=0,
                help="Customer billing agreement duration.",
            )
        with form_r1_c2:
            input_tenure = st.slider(
                "Tenure (Months with Company)",
                min_value=0,
                max_value=72,
                value=6,
                step=1,
                help="Number of months customer has stayed with the company.",
            )
        with form_r1_c3:
            input_charges = st.number_input(
                "Monthly Charges ($ USD)",
                min_value=18.0,
                max_value=125.0,
                value=85.0,
                step=1.0,
                help="Current monthly subscription charge.",
            )

        form_r2_c1, form_r2_c2, form_r2_c3 = st.columns(3)
        with form_r2_c1:
            input_internet = st.selectbox(
                "Internet Service",
                options=["Fiber optic", "DSL", "No"],
                index=0,
                help="Type of internet connection.",
            )
        with form_r2_c2:
            input_payment = st.selectbox(
                "Payment Method",
                options=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                index=0,
                help="Billing payment channel.",
            )
        with form_r2_c3:
            input_support = st.selectbox(
                "Tech Support Add-on",
                options=["No", "Yes", "No internet service"],
                index=0,
                help="Whether subscriber has dedicated tech support.",
            )

        form_r3_c1, form_r3_c2 = st.columns(2)
        with form_r3_c1:
            input_billing = st.selectbox(
                "Paperless Billing",
                options=["Yes", "No"],
                index=0,
                help="Whether paperless e-billing is active.",
            )
        with form_r3_c2:
            input_senior = st.radio(
                "Senior Citizen Status",
                options=["Non-Senior (<65)", "Senior Citizen (65+)"],
                index=0,
                horizontal=True,
            )

        submit_btn = st.form_submit_button("🔍 Evaluate Customer Churn Risk", use_container_width=True)

    if submit_btn:
        customer_dict = {
            "Contract": input_contract,
            "tenure": int(input_tenure),
            "MonthlyCharges": float(input_charges),
            "InternetService": input_internet,
            "PaymentMethod": input_payment,
            "TechSupport": input_support,
            "PaperlessBilling": input_billing,
            "SeniorCitizen": 1 if "65+" in input_senior else 0,
        }

        try:
            pred_outcome = predict_customer_churn_risk(model_results["pipeline"], customer_dict)

            is_at_risk = pred_outcome["predicted_class"] == 1
            card_border = "#ef4444" if is_at_risk else "#10b981"
            badge_bg = "#fee2e2" if is_at_risk else "#dcfce7"
            badge_fg = "#991b1b" if is_at_risk else "#166534"
            status_text = "⚠️ High Risk of Churn" if is_at_risk else "✅ Likely to Retain"

            risk_tier_escaped = html.escape(str(pred_outcome["risk_tier"]))
            contract_escaped = html.escape(str(input_contract))
            rec_escaped = html.escape(str(pred_outcome["recommendation"]))
            disclaimer_escaped = html.escape(str(pred_outcome["disclaimer"]))
            pred_card_html = (
                f'<div class="insight-card" style="border-left: 5px solid {card_border}; margin-top: 1rem;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">'
                f'<span style="font-weight: 700; font-size: 1.25rem; color: #0f172a;">{status_text}</span>'
                f'<span style="background-color: {badge_bg}; color: {badge_fg}; padding: 0.25rem 0.75rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem;">'
                f'{risk_tier_escaped}'
                f'</span>'
                f'</div>'
                f'<div style="display: flex; gap: 2rem; margin: 1rem 0; padding: 0.8rem 1rem; background-color: #f8fafc; border-radius: 8px;">'
                f'<div>'
                f'<div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Predicted Class</div>'
                f'<div style="font-size: 1.2rem; font-weight: 700; color: #0f172a;">Class {pred_outcome["predicted_class"]} ({"Churn" if is_at_risk else "No Churn"})</div>'
                f'</div>'
                f'<div>'
                f'<div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Leaf Churn Proportion</div>'
                f'<div style="font-size: 1.2rem; font-weight: 700; color: {card_border};">{pred_outcome["risk_score_pct"]:.1f}%</div>'
                f'</div>'
                f'<div>'
                f'<div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 600;">Contract & Tenure</div>'
                f'<div style="font-size: 1.2rem; font-weight: 700; color: #0f172a;">{contract_escaped} ({input_tenure} mos)</div>'
                f'</div>'
                f'</div>'
                f'<div class="insight-section-label">💡 Recommended Action</div>'
                f'<div class="insight-text">{rec_escaped}</div>'
                f'<div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.6rem; border-top: 1px solid #f1f5f9; padding-top: 0.4rem;">'
                f'<i>{disclaimer_escaped}</i>'
                f'</div>'
                f'</div>'
            )
            if hasattr(st, "html"):
                st.html(pred_card_html)
            else:
                st.markdown(pred_card_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error evaluating prediction: {e}")

    # 4. Data-Backed Business Insights from the Decision Tree
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Data-Backed Business Insights (Classification Model)", divider="blue")
    st.caption(
        "Insights computed from the trained Decision Tree's feature importances, confusion matrix, "
        "and baseline comparison. Each insight explicitly states what cannot be concluded."
    )

    model_insights = generate_model_insights(model_results)
    if not model_insights:
        st.info("Model results insufficient to generate insights.")
    else:
        mi_col1, mi_col2 = st.columns(2)
        for idx, insight in enumerate(model_insights):
            target_col = mi_col1 if idx % 2 == 0 else mi_col2
            with target_col:
                render_insight_card(insight)

    # 5. Interview Technical Concepts (Plain-Language Guide)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📚 Analytical note: Core Machine Learning Concepts Explained"):
        st.markdown(
            r"""
            ### 1. Overfitting and Tree Depth
            - **Overfitting**: Occurs when a decision tree grows deep and complex, memorizing random noise in the training set rather than underlying patterns. Such a tree has near-perfect training accuracy but performs poorly on unseen test data.
            - **Regularization via `max_depth`**: By capping tree depth (e.g. at 5), we constrain tree complexity, enforcing simpler, more generalizable decision boundaries.

            ### 2. Gini Impurity
            - **Definition**: A measure of label disorder/heterogeneity at a tree node:
              $$Gini = 1 - \sum_{i=1}^{C} p_i^2$$
            - A pure node (100% Churn or 100% Retained) has $Gini = 0$.
            - At each step, the algorithm chooses the split that maximizes the **reduction in Gini impurity**.

            ### 3. Precision, Recall, and F1-Score
            - **Precision**: $\\frac{TP}{TP + FP}$ — Out of all customers the model predicted would churn, what percentage actually left? High precision avoids wasting retention offers on customers who were never going to cancel.
            - **Recall**: $\\frac{TP}{TP + FN}$ — Out of all customers who actually churned, what percentage did the model catch? High recall ensures high-risk accounts don't slip through unnoticed.
            - **F1-Score**: The harmonic mean of precision and recall ($2 \\cdot \\frac{P \\cdot R}{P + R}$), providing a single balanced metric when classes are uneven.

            ### 4. Class Imbalance and The Accuracy Paradox
            - In Telco churn, ~73.5% of customers stay. A trivial model predicting *"everyone stays"* scores 73.5% accuracy but is commercially useless (detects 0 churners). 
            - Therefore, model evaluation must emphasize **Precision, Recall, F1-score, and Lift over the majority baseline**, rather than accuracy alone.

            ### 5. Preventing Data Leakage
            - **Data Leakage**: Happens when information from outside the training dataset (such as test set statistics or post-event target indicators) leaks into model training.
            - **Our Prevention Safeguards**:
              1. 80/20 train/test split was performed **before** any model selection.
              2. Categorical encoders were fitted **only on `X_train`** and applied to `X_test`.
              3. Hyperparameter tuning (`max_depth`) used **5-fold cross-validation on `X_train` exclusively**.
              4. `X_test` was evaluated exactly once.
            """
        )

