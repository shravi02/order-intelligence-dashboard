import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Logistics Intelligence System",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================
DEFAULT_FILE = "data.csv"

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(DEFAULT_FILE)

# =====================================================
# DATA CLEANING
# =====================================================
df.columns = df.columns.str.strip()

numeric_cols = [
    'delay',
    'attempts',
    'order_value'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

df = df.dropna(subset=['delay'])

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 20px;
}

[data-testid="metric-container"] {
    background: white;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #e2e8f0;
}

.stPlotlyChart {
    background: white;
    border-radius: 16px;
    padding: 10px;
}

.insight-box {
    background: white;
    padding: 15px;
    border-radius: 14px;
    border-left: 6px solid #2563eb;
    margin-bottom: 12px;
    font-size: 16px;
}

@media (max-width: 768px) {

    .main-title {
        font-size: 28px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <div class="main-title">
        Logistics Intelligence System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================
section = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Business Health",
        "ML Prediction",
        "Download Reports",
        "Upload Dataset"
    ]
)

# =====================================================
# FILTERS
# =====================================================
st.sidebar.markdown("---")

st.sidebar.subheader("Filters")

payment = st.sidebar.multiselect(
    "Payment Type",
    options=sorted(df['payment_type'].unique()),
    default=sorted(df['payment_type'].unique())
)

risk = st.sidebar.multiselect(
    "Risk Level",
    options=sorted(df['risk'].unique()),
    default=sorted(df['risk'].unique())
)

filtered_df = df[
    (df['payment_type'].isin(payment)) &
    (df['risk'].isin(risk))
]

# =====================================================
# DASHBOARD
# =====================================================
if section == "Dashboard":

    st.header("Executive Summary")

    total_orders = len(filtered_df)

    returned_orders = len(
        filtered_df[
            filtered_df['status'] == 'Returned'
        ]
    )

    revenue = filtered_df['order_value'].sum()

    revenue_loss = filtered_df[
        filtered_df['status'] == 'Returned'
    ]['order_value'].sum()

    rto = (
        returned_orders / total_orders
    ) * 100

    high_risk_orders = len(
        filtered_df[
            filtered_df['risk'] == 'High'
        ]
    )

    avg_delay = filtered_df['delay'].mean()

    k1, k2, k3 = st.columns(3)

    k4, k5, k6 = st.columns(3)

    k1.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

    k2.metric(
        "Revenue",
        f"₹{revenue:,.0f}"
    )

    k3.metric(
        "Returned Orders",
        f"{returned_orders:,}"
    )

    k4.metric(
        "RTO %",
        f"{rto:.2f}%"
    )

    k5.metric(
        "Revenue Loss",
        f"₹{revenue_loss:,.0f}"
    )

    k6.metric(
        "High Risk Orders",
        f"{high_risk_orders:,}"
    )

    st.markdown("---")

    # =================================================
    # SMART INSIGHTS
    # =================================================

    st.subheader("Smart Insights")

    st.markdown("""
    <div class="insight-box">
    ⚠ COD orders show highest return probability
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    ⚠ Orders delayed beyond 7 days are most likely to return
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    ✅ UPI orders have lowest RTO percentage
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # =================================================
    # CHARTS
    # =================================================

    c1, c2 = st.columns(2)

    with c1:

        city_revenue = filtered_df.groupby(
            'city'
        )['order_value'].sum().reset_index()

        fig1 = px.bar(
            city_revenue,
            x='city',
            y='order_value',
            title='Revenue by City',
            color='order_value'
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with c2:

        product_returns = filtered_df[
            filtered_df['status'] == 'Returned'
        ]

        product_returns = product_returns.groupby(
            'product'
        ).size().reset_index(name='returns')

        fig2 = px.bar(
            product_returns,
            x='product',
            y='returns',
            title='Top Returning Products',
            color='returns'
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.markdown("---")

    # =================================================
    # ACTION CENTER
    # =================================================

    st.subheader("Action Center")

    high_risk_df = filtered_df[
        filtered_df['risk'] == 'High'
    ]

    high_risk_df = high_risk_df.sort_values(
        by='delay',
        ascending=False
    )

    st.dataframe(
        high_risk_df[
            [
                'order_id',
                'city',
                'product',
                'delay',
                'status',
                'action'
            ]
        ].head(20),
        use_container_width=True
    )

# =====================================================
# BUSINESS HEALTH
# =====================================================
elif section == "Business Health":

    st.header("Business Health Analytics")

    c1, c2 = st.columns(2)

    # ================================================
    # PAYMENT PERFORMANCE
    # ================================================

    with c1:

        fig3 = px.histogram(
            filtered_df,
            x='payment_type',
            color='status',
            barmode='group',
            title='Payment Method Performance'
        )

        fig3.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # ================================================
    # DELAY ANALYTICS
    # ================================================

    with c2:

        fig4 = px.box(
            filtered_df,
            x='risk',
            y='delay',
            color='risk',
            title='Delay Distribution by Risk'
        )

        fig4.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    st.markdown("---")

    # ================================================
    # ORDER TREND
    # ================================================

    monthly_orders = filtered_df.groupby(
        'order_date'
    ).size().reset_index(name='orders')

    fig5 = px.line(
        monthly_orders,
        x='order_date',
        y='orders',
        title='Order Trend Over Time'
    )

    fig5.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    st.markdown("---")

    # ================================================
    # GEOGRAPHIC ANALYTICS
    # ================================================

    st.subheader("Geographic Analytics")

    city_rto = filtered_df.groupby(
        "city"
    ).apply(
        lambda x: (
            (
                x["status"] == "Returned"
            ).mean()
        ) * 100
    ).reset_index(name="RTO_Percentage")

    city_orders = filtered_df.groupby(
        "city"
    ).size().reset_index(name="Total_Orders")

    city_revenue = filtered_df.groupby(
        "city"
    )['order_value'].sum().reset_index(name="Revenue")

    city_summary = pd.merge(
        city_rto,
        city_orders,
        on="city"
    )

    city_summary = pd.merge(
        city_summary,
        city_revenue,
        on="city"
    )

    # ================================================
    # CITY RISK BUBBLE CHART
    # ================================================

    fig6 = px.scatter(
        city_summary,
        x="city",
        y="RTO_Percentage",
        size="Total_Orders",
        color="Revenue",
        hover_name="city",
        title="City-wise RTO Risk Analysis",
        size_max=60
    )

    fig6.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

    st.markdown("---")

    # ================================================
    # TOP RISKY CITIES
    # ================================================

    st.subheader("Top Risky Cities")

    risky_cities = city_summary.sort_values(
        by="RTO_Percentage",
        ascending=False
    )

    st.dataframe(
        risky_cities,
        use_container_width=True
    )

    st.markdown("---")

    # ================================================
    # PRODUCT RETURN ANALYTICS
    # ================================================

    st.subheader("Top Returning Products")

    product_returns = filtered_df[
        filtered_df['status'] == 'Returned'
    ]

    product_returns = product_returns.groupby(
        'product'
    ).size().reset_index(name='Returns')

    fig7 = px.bar(
        product_returns,
        x='product',
        y='Returns',
        color='Returns',
        title='Product-wise Return Volume'
    )

    fig7.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )

# =====================================================
# ML PREDICTION
# =====================================================
elif section == "ML Prediction":

    st.header("AI Return Risk Prediction")

    # ================================================
    # PREPARE DATA
    # ================================================

    model_df = filtered_df.copy()

    model_df['target'] = model_df['status'].apply(
        lambda x: 1 if x == 'Returned' else 0
    )

    # Encode categorical columns
    model_df = pd.get_dummies(
        model_df,
        columns=[
            'payment_type',
            'city',
            'product_type'
        ]
    )

    feature_columns = [
        col for col in model_df.columns
        if col not in [
            'status',
            'target',
            'order_id',
            'customer_name',
            'product',
            'action',
            'risk',
            'order_date'
        ]
    ]

    X = model_df[feature_columns]

    y = model_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ================================================
    # MODELS
    # ================================================

    rf_model = RandomForestClassifier()

    lr_model = LogisticRegression(
        max_iter=1000
    )

    xgb_model = XGBClassifier(
        eval_metric='logloss'
    )

    # Train models
    rf_model.fit(X_train, y_train)

    lr_model.fit(X_train, y_train)

    xgb_model.fit(X_train, y_train)

    # Predictions
    rf_pred = rf_model.predict(X_test)

    lr_pred = lr_model.predict(X_test)

    xgb_pred = xgb_model.predict(X_test)

    # Accuracy
    rf_acc = accuracy_score(
        y_test,
        rf_pred
    )

    lr_acc = accuracy_score(
        y_test,
        lr_pred
    )

    xgb_acc = accuracy_score(
        y_test,
        xgb_pred
    )

    # ================================================
    # MODEL COMPARISON
    # ================================================

    st.subheader("Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": [
            "Random Forest",
            "Logistic Regression",
            "XGBoost"
        ],
        "Accuracy": [
            rf_acc,
            lr_acc,
            xgb_acc
        ]
    })

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.markdown("---")

    # ================================================
    # CONFUSION MATRIX
    # ================================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        rf_pred
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
    )

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Actual")

    st.pyplot(fig)

    st.markdown("---")

    # ================================================
    # ROC CURVE
    # ================================================

    st.subheader("ROC Curve")

    rf_probs = rf_model.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        rf_probs
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    fig2, ax2 = plt.subplots()

    ax2.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.2f}"
    )

    ax2.plot(
        [0, 1],
        [0, 1],
        linestyle='--'
    )

    ax2.set_xlabel(
        "False Positive Rate"
    )

    ax2.set_ylabel(
        "True Positive Rate"
    )

    ax2.legend()

    st.pyplot(fig2)

    st.markdown("---")

    # ================================================
    # AI RISK SIMULATOR
    # ================================================

    st.subheader("AI Risk Simulator")

    delay_input = st.slider(
        "Delivery Delay",
        0,
        15,
        5
    )

    attempts_input = st.slider(
        "Delivery Attempts",
        1,
        5,
        2
    )

    order_value_input = st.slider(
        "Order Value",
        200,
        5000,
        1500
    )

    if st.button(
        "Predict Return Risk"
    ):

        input_data = X.iloc[0:1].copy()

        input_data[:] = 0

        input_data['delay'] = delay_input

        input_data['attempts'] = attempts_input

        input_data['order_value'] = order_value_input

        result = rf_model.predict(
            input_data
        )[0]

        if result == 1:

            st.error(
                "High Return Risk Predicted"
            )

            st.warning(
                "Recommended Action: Call Customer"
            )

        else:

            st.success(
                "Low Return Risk Predicted"
            )

            st.info(
                "Recommended Action: Normal Delivery"
            )
# =====================================================
# DOWNLOAD REPORTS
# =====================================================
elif section == "Download Reports":

    st.header("Download Reports")

    csv = filtered_df.to_csv(index=False)

    st.download_button(
        label="Download Full Analytics Report",
        data=csv,
        file_name="analytics_report.csv",
        mime="text/csv"
    )

    high_risk_csv = filtered_df[
        filtered_df['risk'] == 'High'
    ].to_csv(index=False)

    st.download_button(
        label="Download High Risk Orders",
        data=high_risk_csv,
        file_name="high_risk_orders.csv",
        mime="text/csv"
    )

# =====================================================
# UPLOAD DATASET
# =====================================================
elif section == "Upload Dataset":

    st.header("Upload Dataset")

    st.write(
        "Upload a CSV dataset for analytics."
    )

    if uploaded_file is not None:

        st.success(
            "Dataset uploaded successfully"
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        st.write(
            "Rows:",
            df.shape[0]
        )

        st.write(
            "Columns:",
            df.shape[1]
        )

    else:

        st.info(
            "Currently using default dataset"
        )