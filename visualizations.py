"""
ChurnLens: Data Visualizations Module
Creates clean, interactive, and beautifully styled Plotly charts for the Streamlit dashboard.
"""

import html
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from churnlens import get_churn_rate_by_category


# Curated color palette
COLOR_CHURN = "#ef4444"      # High contrast coral-red for churned customers
COLOR_RETAIN = "#3b82f6"     # Modern sapphire blue for retained customers
COLOR_NEUTRAL = "#64748b"    # Slate grey for subtle reference lines
COLOR_GRADIENT = ["#3b82f6", "#f59e0b", "#ef4444"]


def create_category_churn_chart(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    horizontal: bool = False
) -> go.Figure:
    """
    Creates an interactive bar chart showing churn rates and customer counts
    by category (Contract, InternetService, PaymentMethod, etc.).

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned or filtered dataset.
    category_col : str
        Column name to group by.
    title : str
        Chart header title.
    horizontal : bool
        If True, renders horizontal bars (ideal for long labels like PaymentMethod).

    Returns:
    --------
    go.Figure:
        Styled Plotly figure object.
    """
    data = get_churn_rate_by_category(df, category_col=category_col, sort_by="churn_rate_desc")

    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected filters",
            showarrow=False,
            font=dict(size=14, color=COLOR_NEUTRAL)
        )
        return fig

    # Custom hover template
    custom_hover = (
        "<b>%{x}</b><br>"
        "Churn Rate: <b>%{y:.1f}%</b><br>"
        "Churned: %{customdata[0]:,} / %{customdata[1]:,} customers<br>"
        "Share of Base: %{customdata[2]:.1f}%"
        "<extra></extra>"
    ) if not horizontal else (
        "<b>%{y}</b><br>"
        "Churn Rate: <b>%{x:.1f}%</b><br>"
        "Churned: %{customdata[0]:,} / %{customdata[1]:,} customers<br>"
        "Share of Base: %{customdata[2]:.1f}%"
        "<extra></extra>"
    )

    customdata = data[["Churned Customers", "Total Customers", "Share of Base (%)"]].values

    if horizontal:
        fig = go.Figure(
            go.Bar(
                x=data["Churn Rate (%)"],
                y=data[category_col],
                orientation="h",
                text=data["Churn Rate (%)"].apply(lambda v: f" {v:.1f}%"),
                textposition="outside",
                marker=dict(
                    color=data["Churn Rate (%)"],
                    colorscale="Reds",
                    cmin=0,
                    cmax=60,
                    line=dict(color="rgba(0,0,0,0.1)", width=1),
                ),
                customdata=customdata,
                hovertemplate=custom_hover,
            )
        )
        fig.update_layout(
            xaxis=dict(
                title="Churn Rate (%)",
                range=[0, max(data["Churn Rate (%)"].max() * 1.18, 10)],
                ticksuffix="%",
                gridcolor="rgba(128, 128, 128, 0.15)",
            ),
            yaxis=dict(
                title="",
                autorange="reversed",
            ),
        )
    else:
        fig = go.Figure(
            go.Bar(
                x=data[category_col],
                y=data["Churn Rate (%)"],
                text=data["Churn Rate (%)"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
                marker=dict(
                    color=data["Churn Rate (%)"],
                    colorscale="Reds",
                    cmin=0,
                    cmax=60,
                    line=dict(color="rgba(0,0,0,0.1)", width=1),
                ),
                customdata=customdata,
                hovertemplate=custom_hover,
            )
        )
        fig.update_layout(
            xaxis=dict(
                title="",
                gridcolor="rgba(128, 128, 128, 0.15)",
            ),
            yaxis=dict(
                title="Churn Rate (%)",
                range=[0, max(data["Churn Rate (%)"].max() * 1.18, 10)],
                ticksuffix="%",
                gridcolor="rgba(128, 128, 128, 0.15)",
            ),
        )

    # Add overall dataset benchmark line if available
    overall_churn = float((df["Churn"] == "Yes").mean() * 100) if len(df) > 0 else 0.0
    if not horizontal:
        fig.add_hline(
            y=overall_churn,
            line_dash="dot",
            line_color=COLOR_NEUTRAL,
            annotation_text=f"Average ({overall_churn:.1f}%)",
            annotation_position="top right",
            annotation_font=dict(size=10, color=COLOR_NEUTRAL),
        )
    else:
        fig.add_vline(
            x=overall_churn,
            line_dash="dot",
            line_color=COLOR_NEUTRAL,
            annotation_text=f"Average ({overall_churn:.1f}%)",
            annotation_position="top right",
            annotation_font=dict(size=10, color=COLOR_NEUTRAL),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, weight="bold")),
        margin=dict(l=20, r=20, t=50, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=340,
    )

    return fig


def create_tenure_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Creates an overlaid tenure distribution and box plot comparing
    churned vs retained customers.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned or filtered dataset.

    Returns:
    --------
    go.Figure:
        Interactive Plotly figure.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for the selected filters",
            showarrow=False,
            font=dict(size=14, color=COLOR_NEUTRAL)
        )
        return fig

    churned = df[df["Churn"] == "Yes"]
    retained = df[df["Churn"] == "No"]

    churned_median = float(churned["tenure"].median()) if not churned.empty else 0.0
    retained_median = float(retained["tenure"].median()) if not retained.empty else 0.0

    fig = go.Figure()

    # Retained Customers Histogram
    fig.add_trace(
        go.Histogram(
            x=retained["tenure"],
            name="Retained (No Churn)",
            marker_color=COLOR_RETAIN,
            opacity=0.65,
            xbins=dict(start=0, end=72, size=3),
            hovertemplate="Retained: <b>%{y}</b> customers<br>Tenure: %{x} months<extra></extra>",
        )
    )

    # Churned Customers Histogram
    fig.add_trace(
        go.Histogram(
            x=churned["tenure"],
            name="Churned (Yes)",
            marker_color=COLOR_CHURN,
            opacity=0.75,
            xbins=dict(start=0, end=72, size=3),
            hovertemplate="Churned: <b>%{y}</b> customers<br>Tenure: %{x} months<extra></extra>",
        )
    )

    # Add vertical indicators for median tenure
    fig.add_vline(
        x=churned_median,
        line_dash="dash",
        line_color=COLOR_CHURN,
        line_width=2,
        annotation_text=f"Median Churned: {churned_median:.0f} mo",
        annotation_position="top left",
        annotation_font=dict(size=11, color=COLOR_CHURN, weight="bold"),
    )
    fig.add_vline(
        x=retained_median,
        line_dash="dash",
        line_color=COLOR_RETAIN,
        line_width=2,
        annotation_text=f"Median Retained: {retained_median:.0f} mo",
        annotation_position="top right",
        annotation_font=dict(size=11, color=COLOR_RETAIN, weight="bold"),
    )

    fig.update_layout(
        title=dict(
            text="Customer Tenure Distribution by Churn Status",
            font=dict(size=16, weight="bold")
        ),
        barmode="overlay",
        xaxis=dict(
            title="Tenure (Months with Company)",
            dtick=12,
            range=[0, 75],
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        yaxis=dict(
            title="Customer Count",
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=30),
        height=360,
    )

    return fig


# Curated palette for up to 7 distinct customer clusters
CLUSTER_PALETTE = [
    "#2563eb",  # Vibrant Blue
    "#ef4444",  # Crimson Red
    "#10b981",  # Emerald Green
    "#f59e0b",  # Amber Orange
    "#8b5cf6",  # Violet Purple
    "#ec4899",  # Pink
    "#06b6d4",  # Cyan
]


def create_elbow_and_silhouette_chart(
    eval_results: dict,
    selected_k: int = 4
) -> go.Figure:
    """
    Creates a side-by-side diagnostic figure displaying:
    1. The Elbow Method (Inertia vs Candidate k)
    2. Silhouette Score analysis across candidate k values

    Parameters:
    -----------
    eval_results : dict
        Output from churnlens.evaluate_kmeans_k_range().
    selected_k : int
        The currently selected or active k value to highlight.

    Returns:
    --------
    go.Figure:
        Plotly figure with two subplots.
    """
    k_vals = eval_results["k_values"]
    inertias = eval_results["inertias"]
    sil_scores = eval_results["silhouette_scores"]
    recommended_k = eval_results["recommended_k"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "<b>Elbow Method (Inertia vs. k)</b>",
            "<b>Silhouette Score Analysis (Cohesion vs. k)</b>",
        ),
        horizontal_spacing=0.12,
    )

    # 1. Elbow Line
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=inertias,
            mode="lines+markers",
            name="Inertia (WCSS)",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=8, color="#1e40af"),
            hovertemplate="k = %{x}<br>Inertia: <b>%{y:,.1f}</b><extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Highlight selected k on Elbow plot
    sel_inertia = inertias[k_vals.index(selected_k)] if selected_k in k_vals else inertias[0]
    fig.add_trace(
        go.Scatter(
            x=[selected_k],
            y=[sel_inertia],
            mode="markers",
            name=f"Selected k={selected_k}",
            marker=dict(size=14, color="#ef4444", symbol="star", line=dict(color="#ffffff", width=1.5)),
            hovertemplate=f"Active k={selected_k}<br>Inertia: <b>{sel_inertia:,.1f}</b><extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2. Silhouette Line
    fig.add_trace(
        go.Scatter(
            x=k_vals,
            y=sil_scores,
            mode="lines+markers",
            name="Silhouette Score",
            line=dict(color="#10b981", width=2.5),
            marker=dict(size=8, color="#047857"),
            hovertemplate="k = %{x}<br>Silhouette Score: <b>%{y:.4f}</b><extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Highlight selected k on Silhouette plot
    sel_sil = sil_scores[k_vals.index(selected_k)] if selected_k in k_vals else sil_scores[0]
    fig.add_trace(
        go.Scatter(
            x=[selected_k],
            y=[sel_sil],
            mode="markers",
            name=f"Selected k={selected_k}",
            showlegend=False,
            marker=dict(size=14, color="#ef4444", symbol="star", line=dict(color="#ffffff", width=1.5)),
            hovertemplate=f"Active k={selected_k}<br>Silhouette: <b>{sel_sil:.4f}</b><extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Annotate peak silhouette if it differs or matches
    peak_sil = max(sil_scores)
    peak_k = k_vals[sil_scores.index(peak_sil)]
    fig.add_annotation(
        x=peak_k,
        y=peak_sil,
        text=f"Peak: k={peak_k} ({peak_sil:.4f})",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#047857",
        ax=0,
        ay=-25,
        font=dict(size=10, color="#047857", weight="bold"),
        row=1,
        col=2,
    )

    fig.update_xaxes(
        title_text="Number of Clusters (k)",
        tickmode="linear",
        dtick=1,
        gridcolor="rgba(128, 128, 128, 0.15)",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Number of Clusters (k)",
        tickmode="linear",
        dtick=1,
        gridcolor="rgba(128, 128, 128, 0.15)",
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title_text="Inertia (Lower = More Compact)",
        gridcolor="rgba(128, 128, 128, 0.15)",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Silhouette Score (Higher = Better Separation)",
        gridcolor="rgba(128, 128, 128, 0.15)",
        row=1,
        col=2,
    )

    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )

    return fig


def create_cluster_scatter_plot(
    df: pd.DataFrame,
    centroids: list,
    x_col: str = "tenure",
    y_col: str = "MonthlyCharges",
) -> go.Figure:
    """
    Creates an interactive scatter plot of customer clusters with centroid indicators.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing 'Cluster', x_col, and y_col.
    centroids : list of dict
        Centroid coordinates in original feature units.
    x_col : str
        Feature name on X-axis (default: 'tenure').
    y_col : str
        Feature name on Y-axis (default: 'MonthlyCharges').

    Returns:
    --------
    go.Figure:
        Interactive Plotly figure.
    """
    if "Cluster" not in df.columns:
        raise KeyError("DataFrame must contain a 'Cluster' column.")

    if isinstance(centroids, dict) and "centroids" in centroids:
        centroids = centroids["centroids"]

    fig = go.Figure()

    unique_clusters = sorted(df["Cluster"].unique())

    # Add points for each cluster
    for c_id in unique_clusters:
        c_df = df[df["Cluster"] == c_id]
        color = CLUSTER_PALETTE[c_id % len(CLUSTER_PALETTE)]

        hover = (
            f"<b>Cluster {c_id}</b><br>"
            "Customer ID: %{customdata[0]}<br>"
            "Tenure: <b>%{x} months</b><br>"
            "Monthly Charges: <b>$%{y:.2f}</b><br>"
            "Churn Status: %{customdata[1]}<extra></extra>"
        )

        customdata = c_df[["customerID", "Churn"]].values

        fig.add_trace(
            go.Scatter(
                x=c_df[x_col],
                y=c_df[y_col],
                mode="markers",
                name=f"Cluster {c_id}",
                marker=dict(
                    color=color,
                    size=5.5,
                    opacity=0.6,
                    line=dict(width=0),
                ),
                customdata=customdata,
                hovertemplate=hover,
            )
        )

    # Add Centroid Markers
    for c_dict in centroids:
        c_id = c_dict["Cluster"]
        c_x = c_dict[x_col]
        c_y = c_dict[y_col]

        fig.add_trace(
            go.Scatter(
                x=[c_x],
                y=[c_y],
                mode="markers+text",
                name=f"Centroid {c_id}",
                text=[f"C{c_id}"],
                textposition="top center",
                textfont=dict(size=11, color="#0f172a", weight="bold"),
                marker=dict(
                    color="#ffffff",
                    size=14,
                    symbol="diamond",
                    line=dict(color="#0f172a", width=2.5),
                ),
                hovertemplate=(
                    f"<b>Centroid {c_id}</b><br>"
                    f"Avg Tenure: <b>{c_x:.1f} months</b><br>"
                    f"Avg Monthly Charges: <b>${c_y:.2f}</b><extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(
            text="Customer Segments: Tenure vs. Monthly Charges",
            font=dict(size=16, weight="bold"),
        ),
        xaxis=dict(
            title="Customer Tenure (Months)",
            range=[-1, 75],
            dtick=12,
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        yaxis=dict(
            title="Monthly Charges ($ USD)",
            range=[10, 128],
            ticksuffix="$",
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=30),
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def create_confusion_matrix_chart(cm_data: dict) -> go.Figure:
    """
    Creates an interactive annotated heatmap for the test set confusion matrix.

    Parameters:
    -----------
    cm_data : dict
        Contains 'tn', 'fp', 'fn', 'tp', 'total_test'.

    Returns:
    --------
    go.Figure:
        Interactive Plotly heatmap.
    """
    tn = cm_data["tn"]
    fp = cm_data["fp"]
    fn = cm_data["fn"]
    tp = cm_data["tp"]
    total = cm_data["total_test"]

    z_values = [[tn, fp], [fn, tp]]
    x_labels = ["Predicted: Retained (0)", "Predicted: Churned (1)"]
    y_labels = ["Actual: Retained (0)", "Actual: Churned (1)"]

    # Descriptive cell texts
    text_annotations = [
        [
            f"<b>True Negative (TN)</b><br>{tn:,} customers<br>({tn/total*100:.1f}%)",
            f"<b>False Positive (FP)</b><br>{fp:,} customers<br>({fp/total*100:.1f}%)",
        ],
        [
            f"<b>False Negative (FN)</b><br>{fn:,} customers<br>({fn/total*100:.1f}%)",
            f"<b>True Positive (TP)</b><br>{tp:,} customers<br>({tp/total*100:.1f}%)",
        ],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=text_annotations,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#0f172a"),
            colorscale=[
                [0.0, "#eff6ff"],
                [0.5, "#93c5fd"],
                [1.0, "#3b82f6"],
            ],
            showscale=False,
            hovertemplate=(
                "%{y} | %{x}<br>Count: <b>%{z:,}</b><extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Test Set Confusion Matrix (n = {total:,})",
            font=dict(size=15, weight="bold"),
        ),
        xaxis=dict(side="bottom", tickfont=dict(size=11, weight="bold")),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, weight="bold")),
        margin=dict(l=20, r=20, t=50, b=30),
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_feature_importance_chart(fi_df: pd.DataFrame, top_n: int = 8) -> go.Figure:
    """
    Creates an interactive horizontal bar chart for Decision Tree feature importances.

    Parameters:
    -----------
    fi_df : pd.DataFrame
        Table with 'feature' and 'importance' columns.
    top_n : int
        Number of top features to render.

    Returns:
    --------
    go.Figure:
        Plotly horizontal bar chart.
    """
    plot_df = fi_df.head(top_n).sort_values(by="importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=plot_df["importance"] * 100,
            y=plot_df["feature"],
            orientation="h",
            text=plot_df["importance"].apply(lambda v: f" {v * 100:.1f}%"),
            textposition="outside",
            marker=dict(
                color=plot_df["importance"],
                colorscale="Blues",
                cmin=0,
                cmax=max(plot_df["importance"].max(), 0.1),
                line=dict(color="rgba(0,0,0,0.1)", width=1),
            ),
            hovertemplate=(
                "<b>%{y}</b><br>Relative Model Reliance: <b>%{x:.2f}%</b><extra></extra>"
            ),
        )
    )

    max_x = max(plot_df["importance"].max() * 100 * 1.25, 15)

    fig.update_layout(
        title=dict(
            text=f"Top {len(plot_df)} Predictive Features (Gini Impurity Reduction)",
            font=dict(size=15, weight="bold"),
        ),
        xaxis=dict(
            title="Relative Importance (%)",
            ticksuffix="%",
            range=[0, max_x],
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        yaxis=dict(title=""),
        margin=dict(l=20, r=20, t=50, b=30),
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_depth_cv_chart(cv_df: pd.DataFrame, selected_depth: int) -> go.Figure:
    """
    Plots training cross-validation F1 score across candidate tree depths.

    Parameters:
    -----------
    cv_df : pd.DataFrame
        Cross-validation results table.
    selected_depth : int
        Depth selected by CV.

    Returns:
    --------
    go.Figure:
        Line chart.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cv_df["Depth"],
            y=cv_df["Mean CV F1"],
            mode="lines+markers",
            name="Mean 5-Fold CV F1",
            line=dict(color="#2563eb", width=2.5),
            marker=dict(size=8, color="#1e40af"),
            hovertemplate="Tree Depth: %{x}<br>Mean CV F1: <b>%{y:.4f}</b><extra></extra>",
        )
    )

    # Highlight selected depth
    sel_row = cv_df[cv_df["Depth"] == selected_depth]
    if not sel_row.empty:
        fig.add_trace(
            go.Scatter(
                x=[selected_depth],
                y=[float(sel_row["Mean CV F1"].iloc[0])],
                mode="markers",
                name=f"Selected Depth = {selected_depth}",
                marker=dict(size=14, color="#ef4444", symbol="star", line=dict(color="#ffffff", width=1.5)),
                hovertemplate=f"Selected Depth: {selected_depth}<br>F1: <b>{float(sel_row['Mean CV F1'].iloc[0]):.4f}</b><extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="Training 5-Fold Cross-Validation vs. Tree Depth",
            font=dict(size=14, weight="bold"),
        ),
        xaxis=dict(
            title="Candidate max_depth",
            dtick=1,
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        yaxis=dict(
            title="Churn F1 Score",
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        margin=dict(l=20, r=20, t=40, b=30),
        height=260,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def build_insight_card_html(insight: dict) -> str:
    """
    Constructs an unindented, properly escaped HTML string for a 4-part Business Insight card.

    Technical Safeguards:
    1. Escapes all dynamic text values (title, category, problem, evidence, recommendation, limitation)
       using html.escape to prevent injection or malformed tags.
    2. Uses unindented layout without internal blank lines between block elements, preventing
       Markdown-it / CommonMark parsers from interpreting indented lines as <pre><code> blocks.
    """
    severity = str(insight.get("severity", "Medium"))
    severity_lower = severity.lower()
    card_border = "#ef4444" if severity_lower == "high" else "#f59e0b" if severity_lower == "medium" else "#10b981"
    badge_bg = "#fee2e2" if severity_lower == "high" else "#fef3c7" if severity_lower == "medium" else "#dcfce7"
    badge_fg = "#991b1b" if severity_lower == "high" else "#92400e" if severity_lower == "medium" else "#166534"

    title = html.escape(str(insight.get("title", "Business Insight")))
    category = html.escape(str(insight.get("category", "Operational")))
    problem = html.escape(str(insight.get("problem", "")))
    evidence = html.escape(str(insight.get("evidence", "")))
    recommendation = html.escape(str(insight.get("recommendation", "")))
    limitation = html.escape(str(insight.get("limitation", "")))
    severity_label = html.escape(f"{severity} Priority")

    return (
        f'<div class="insight-card" style="border-left-color: {card_border};">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">'
        f'<div>'
        f'<span class="insight-badge" style="background-color: {badge_bg}; color: {badge_fg};">{severity_label}</span> '
        f'<span class="insight-badge badge-category">{category}</span>'
        f'</div>'
        f'</div>'
        f'<div class="insight-title">{title}</div>'
        f'<div class="insight-section-label" style="color: #b91c1c;">🔴 1. Problem Identified</div>'
        f'<div class="insight-text">{problem}</div>'
        f'<div class="insight-section-label" style="color: #1d4ed8;">📊 2. Measured Evidence</div>'
        f'<div class="insight-text">{evidence}</div>'
        f'<div class="insight-section-label" style="color: #047857;">💡 3. Recommended Investigation</div>'
        f'<div class="insight-text">{recommendation}</div>'
        f'<div class="insight-section-label" style="color: #b45309;">⚠️ 4. Analytical Limitation</div>'
        f'<div class="insight-limitation-box">{limitation}</div>'
        f'</div>'
    )
