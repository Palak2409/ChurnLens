# ChurnLens: Customer Churn Analytics and Risk Predictor

An end-to-end customer churn intelligence dashboard built with Python, Scikit-learn, Plotly, and Streamlit. ChurnLens transforms raw telecom customer records into exploratory analytics, unsupervised customer segmentation, and supervised churn risk predictions accompanied by disciplined, four-part business insights.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset & Data Governance](#dataset--data-governance)
3. [Architecture & Technology Stack](#architecture--technology-stack)
4. [Application Structure](#application-structure)
5. [Methodology & Machine Learning Engineering](#methodology--machine-learning-engineering)
   - [Data Cleaning & Integrity](#data-cleaning--integrity)
   - [Customer Segmentation (K-Means)](#customer-segmentation-k-means)
   - [Churn Risk Classification (Decision Tree)](#churn-risk-classification-decision-tree)
6. [Disciplined Business Insights Framework](#disciplined-business-insights-framework)
7. [Installation & Setup](#installation--setup)
8. [Automated Testing Suite](#automated-testing-suite)
9. [Limitations & Future Improvements](#limitations--future-improvements)

---

## Project Overview

Customer churn poses a significant challenge for subscription-based businesses. Identifying high-risk accounts and understanding churn drivers enables proactive retention rather than reactive damage control.

**ChurnLens** delivers three operational lenses:
1. **Overview**: Exploratory data analysis, cohort filtering, and four-part data-backed insights.
2. **Customer Segments**: Unsupervised clustering (K-Means) on tenure and spending behavior with dynamic persona profiling.
3. **Churn Risk**: Supervised classification (Decision Tree) with cross-validation tuning, comprehensive evaluation metrics against a majority-class baseline, and single-customer risk scoring.

---

## Dataset & Data Governance

- **Source**: IBM Telco Customer Churn dataset (available publicly via Kaggle / IBM Business Analytics Community).
- **Scale**: 7,043 customer accounts across 21 raw attributes.
- **Attributes**: Demographics (gender, partner, dependents, senior citizen status), subscribed services (phone, multiple lines, internet service, online security, tech support, streaming), account details (contract type, billing method, payment method, tenure, monthly charges, total charges), and target label (`Churn`: Yes/No).
- **Location**: Automatically loaded from `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (or project root if placed there).

---

## Architecture & Technology Stack

ChurnLens strictly separates data processing, modeling logic, chart generation, and UI presentation:

```
ChurnLens/
│
├── churnlens.py            # Data loading, validation, cleaning, metrics, K-Means, Decision Tree, insight generation
├── visualizations.py       # Reusable Plotly chart builders (color palettes, styling, annotations)
├── app.py                  # Streamlit user interface across the three tabs
├── requirements.txt        # Pinned dependencies
├── README.md               # Architecture documentation and interview reference
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
└── tests/
    └── test_churnlens.py   # 30 comprehensive pytest unit and integration tests
```

- **Core Runtime**: Python 3.12+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (`StandardScaler`, `KMeans`, `DecisionTreeClassifier`, `Pipeline`, `ColumnTransformer`, `GridSearchCV`, `StratifiedKFold`)
- **Visualizations**: Plotly Graph Objects & Subplots
- **Interface**: Streamlit
- **Quality Assurance**: Pytest

---

## Application Structure

### Tab 1: Overview
- **Global KPI Cards**: Total customer count, overall churn rate (%), average monthly charges ($), and average tenure (months).
- **Interactive Cohort Filters**: Multi-select filtering by Contract, Internet Service, and Payment Method. KPI cards show the filtered cohort.
- **Interactive Visualizations**:
  - Churn rate by Contract Type (Month-to-month vs. One year vs. Two year).
  - Churn rate by Internet Service (Fiber optic vs. DSL vs. No internet).
  - Churn rate by Payment Method (Electronic check vs. Mailed check vs. Bank transfer vs. Credit card).
  - Tenure distributions comparing retained vs. churned customers.
- **Data-Backed Insights**: Four structured insights highlighting contract friction, fiber service churn, payment method vulnerability, and early-lifecycle retention.

### Tab 2: Customer Segments
- **Unsupervised Clustering**: Standardized 2D feature space (`tenure` and `MonthlyCharges`).
- **Cluster Diagnostics**: Interactive Elbow Curve (Inertia vs. k) and Silhouette Score analysis for $k \in [2, 7]$.
- **Cluster Distribution**: Scatter plot of tenure vs. monthly charges colored by assigned cluster, featuring cluster centroids.
- **Automated Persona Generation**: Computes cluster centroids, observed churn rates, customer volumes, and automatically derives human-readable personas (e.g., *Budget Long-Term Loyalist*, *High-Spend Early Flight Risk*).
- **Segmentation Insights**: Highest-churn cluster intervention, lowest-churn stability analysis, and high-spend retention opportunities.

### Tab 3: Churn Risk
- **Supervised Classification**: Interpretable Decision Tree pipeline trained on 8 features.
- **Model Diagnostics**:
  - Training set 5-fold cross-validation performance curve across candidate depths ($d \in [2, 8]$) targeting macro/binary F1-score.
  - Held-out test set performance table comparing Accuracy, Precision, Recall, and F1-Score against the Zero-Rule (majority-class) baseline.
  - Feature Importance bar chart illustrating relative Gini reduction.
  - Confusion Matrix with raw counts, true/false rates, and explicit quadrant explanations (TP, FP, TN, FN).
- **Single-Customer Risk Calculator**: Form allowing interactive customer profile inputs with immediate leaf-node risk estimation, risk tier assignment (Low, Medium, High), and tailored retention guidance.
- **Model Insights & Interview Guide**: Technical trade-off documentation, precision-recall balancing rationale, and interview Q&A reference.

---

## Methodology & Machine Learning Engineering

### Data Cleaning & Integrity

1. **Numeric Conversion of `TotalCharges`**:
   The raw dataset contains 11 customer records with blank string spaces (`" "`) in `TotalCharges`.
   - *Root Cause Analysis*: All 11 records have `tenure == 0` (brand-new accounts signed up within the current billing cycle who have not yet received their first bill).
   - *Engineering Decision*: Rather than discarding these valid customer rows or filling with an arbitrary mean/median, `TotalCharges` is converted to numeric with invalid entries coerced to NaN, and records with `tenure == 0` are safely imputed to `0.0`.
2. **Duplicate Deduplication**:
   Verified uniqueness across `customerID`. Deduplication routines check for and log any redundant entries while preserving the first instance.
3. **Target Encoding**:
   The target `Churn` is mapped to binary integer format (`Yes` $\to 1$, `No` $\to 0$) for scikit-learn estimators, while original descriptive strings are preserved for display and charting.

### Customer Segmentation (K-Means)

1. **Feature Selection**:
   - `tenure` (0 to 72 months)
   - `MonthlyCharges` (~$18.25 to $118.75)
2. **Standardization**:
   Because tenure and monthly charges operate on different numerical scales, Euclidean distance in unscaled space would bias clustering toward the feature with larger variance. Both features are scaled via `StandardScaler` ($\mu = 0, \sigma = 1$).
3. **Cluster Selection ($k$)**:
   - Inertia (Elbow Method) measures within-cluster sum of squares.
   - Silhouette Score evaluates intra-cluster cohesion versus nearest-cluster separation.
   - The optimal $k$ is selected based on maximizing the silhouette coefficient across the tested range ($k=2$ to $k=7$), with $k=4$ consistently exhibiting strong, interpretable separation.
4. **Reproducibility**:
   Fitted with `random_state=42` and `n_init=10` to guarantee deterministic convergence.

### Churn Risk Classification (Decision Tree)

1. **Feature Selection**:
   An 8-feature subset selected for business interpretability and governance:
   - `Contract` (categorical)
   - `tenure` (numeric)
   - `MonthlyCharges` (numeric)
   - `InternetService` (categorical)
   - `PaymentMethod` (categorical)
   - `TechSupport` (categorical)
   - `PaperlessBilling` (categorical)
   - `SeniorCitizen` (binary numeric)
   
   *Exclusions*: `customerID` is excluded as a non-predictive primary key. `TotalCharges` is excluded to avoid collinearity with $\text{tenure} \times \text{MonthlyCharges}$.
2. **Train/Test Split**:
   - 80% Training ($n = 5,634$), 20% Held-Out Testing ($n = 1,409$).
   - Stratified on `Churn` to preserve the 73.5% / 26.5% class balance.
   - Seeded with `random_state=42`.
3. **Pipeline & Unseen Category Defense**:
   Utilizes scikit-learn `Pipeline` and `ColumnTransformer`. Categorical variables pass through `OneHotEncoder(handle_unknown="ignore")` to prevent inference-time failures on unobserved categories.
4. **Model Selection via Cross-Validation**:
   Candidate tree depths ($d \in [2, 7]$) are evaluated strictly on the training set using 5-Fold Stratified Cross-Validation optimizing F1-score (`f1_weighted` / `f1_binary`). The selected depth ($d=2$) balances decision complexity and generalization without test set leakage.
5. **Test Set Evaluation & Baseline Comparison**:
   The selected model is evaluated once on the test set against the Majority Class Baseline (predicting No Churn for all customers):
   - **Baseline Accuracy**: 73.46% (Precision: 0.00%, Recall: 0.00%, F1: 0.00%)
   - **Decision Tree Accuracy**: ~75.09%
   - **Decision Tree Precision**: ~52.7%
   - **Decision Tree Recall**: ~59.4%
   - **Decision Tree F1-Score**: ~0.5585

---

## Disciplined Business Insights Framework

Every insight presented across the application strictly implements a four-part analytical framework designed to uphold rigorous data science standards:

1. **Problem**: Clear, objective articulation of the observed pattern or potential operational friction.
2. **Evidence**: Exact empirical measurements, including observed churn rates, customer volumes, percentage deltas, or model metrics.
3. **Recommendation**: Constructive, non-prescriptive next steps for business stakeholders to explore.
4. **Limitation**: Explicit epistemological boundary stating what *cannot* be concluded (e.g., correlation is not causation, uncalibrated leaf proportions are not Bayesian posterior probabilities, confounding variables may exist).

### Core Analytical Commitments:
- **No Causal Fallacies**: Features like `Month-to-month` contract or `Fiber optic` service are reported as correlated risk indicators, never as direct causes.
- **No Presumptive Outcomes**: Individual accounts are scored by leaf-node historical proportions, never labeled as "guaranteed to leave".
- **No Fabricated Financials**: No invented dollar savings, ROI claims, or hypothetical retention multipliers without experimental evidence.

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### Quickstart

1. **Clone or Navigate to the Repository**:
   ```bash
   cd ChurnLens
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Data Placement**:
   Download the IBM Telco Customer Churn dataset from `https://www.kaggle.com/datasets/blastchar/telco-customer-churn` and place it in data/.

5. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## Automated Testing Suite

The project includes an automated test suite verifying data hygiene, model pipelines, visualization contracts, and business insight integrity:

```bash
python -m pytest -v
```

### Test Suite Coverage (32 Tests):
- `test_load_data_returns_dataframe`: CSV loading and return type verification.
- `test_load_data_invalid_path`: Graceful handling of missing files.
- `test_validate_columns`: Verification of required schema columns.
- `test_check_data_quality`: Diagnostic report checks for nulls, types, and duplicates.
- `test_clean_data_preserves_zero_tenure`: Confirms 11 zero-tenure accounts are preserved with imputed 0.0 charges.
- `test_clean_data_removes_duplicates`: Deduplication verification.
- `test_clean_data_target_encoding`: Validates binary target mapping (0 and 1).
- `test_clean_data_on_actual_dataset`: Integration test on the full 7,043 record dataset.
- `test_get_churn_summary`: Overall customer count and churn rate calculation.
- `test_get_charge_and_tenure_metrics`: Charge and tenure arithmetic accuracy.
- `test_get_churn_rate_by_category`: Grouped churn metrics across categorical variables.
- `test_get_tenure_distribution`: Binning and tenure histogram aggregations.
- `test_filter_customers_and_cohort_metrics`: Cohort filtering with delta baseline calculations.
- `test_compute_business_insights`: Verifies 4 overview insights.
- `test_create_category_churn_chart`: Verifies Plotly bar chart generation.
- `test_create_tenure_distribution_chart`: Verifies Plotly histogram generation.
- `test_evaluate_kmeans_k_range`: Evaluates inertia and silhouette score calculations across $k \in [2, 7]$.
- `test_fit_customer_clusters_reproducibility`: Confirms deterministic cluster assignments across identical random seeds.
- `test_fit_customer_clusters_assignment_and_summary`: Validates centroid summaries and column assignments.
- `test_kmeans_visualizations`: Validates scatter plot and elbow curve generation.
- `test_train_decision_tree_model`: Verifies 80/20 stratified split, cross-validation tuning, and test evaluation.
- `test_decision_tree_pipeline_unseen_categories`: Verifies pipeline handles unseen categorical inputs without throwing errors.
- `test_predict_customer_churn_risk`: Single-customer risk scoring, risk tier assignment, and leaf calculation.
- `test_decision_tree_visualizations`: Validates feature importance and confusion matrix charts.
- `test_compute_business_insights_limitation_key`: Checks all 4 required insight keys (`problem`, `evidence`, `recommendation`, `limitation`).
- `test_compute_business_insights_dynamic_sensitivity`: Verifies insight evidence strings update when underlying cohort data changes.
- `test_generate_segment_insights`: Verifies clustering insights structure and required keys.
- `test_generate_segment_insights_dynamic_sensitivity`: Confirms segment insight evidence reacts dynamically to different cluster topologies.
- `test_generate_model_insights`: Verifies model insight structure and required keys.
- `test_generate_model_insights_dynamic_values`: Confirms model insight evidence reflects actual test set confusion matrix metrics.
- `test_build_insight_card_html_structure_and_safety` : Validates that build_insight_card_html generates clean, unindented HTML, includes all four section labels, and safely escapes dangerous characters to prevent XSS and malformed tags.
- `test_build_insight_card_html_on_actual_dataset` : Verifies that all overview insights generated from actual data produce valid HTML cards.
---

## Limitations & Future Improvements

1. **Observational Data Only**:
   The IBM Telco dataset is cross-sectional and observational. A/B testing or uplift modeling would be required before attributing churn reduction to any intervention.
2. **Uncalibrated Leaf Proportions**:
   Decision Tree leaf proportions reflect local sample frequencies, not calibrated Bayesian probabilities. Calibrating probabilities via isotonic regression or Platt scaling is a natural next step.
3. **Linear & Orthogonal Boundaries**:
   Decision Trees use axis-aligned splits, which may sub-optimally capture diagonal decision boundaries. Comparing against Regularized Logistic Regression or Gradient Boosted Trees (XGBoost/LightGBM) would provide further performance benchmarking.
4. **Static Snapshot**:
   The dataset captures a single point in time. Time-series survival analysis (e.g., Cox Proportional Hazards) would enable dynamic churn risk forecasting over customer lifecycles.
