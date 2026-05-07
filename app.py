import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="TintBox Analytics",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("data.csv")

# =====================================================
# DATA CLEANING
# =====================================================
df = df.dropna()

df["delay"] = pd.to_numeric(df["delay"], errors="coerce")
df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce")

df = df.dropna(subset=["delay"])

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

/* =====================================================
MAIN APP
===================================================== */

.stApp {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 100%;
}

/* =====================================================
HEADER
===================================================== */

.header-container {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 40px;
    margin-top: 10px;
    margin-bottom: 25px;
    flex-wrap: nowrap;
}

/* Logo */
.logo-img {
    height: 110px;
    width: auto;
    object-fit: contain;
}

/* Title */
.title-text {
    font-size: 34px;
    font-weight: 700;
    color: #1e293b;
    margin: 0;
    padding: 0;
    text-align: center;
    line-height: 1.1;
}

/* =====================================================
METRIC CARDS
===================================================== */

[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #e5e7eb;
}

/* =====================================================
CHARTS
===================================================== */

.stPlotlyChart {
    background: white;
    border-radius: 14px;
    padding: 10px;
}

/* =====================================================
TABLE
===================================================== */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* =====================================================
MOBILE RESPONSIVE
===================================================== */

@media (max-width: 768px) {

    .header-container {
        flex-direction: column;
        gap: 10px;
    }

    .logo-img {
        height: 85px;
    }

    .title-text {
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
    <div class="header-container">

        <img
            class="logo-img"
            src="https://tintbox.in/cdn/shop/files/TintBox_Logo.png?v=1679057052"
        >

        <div class="title-text">
            TintBox Analytics
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# FILTERS
# =====================================================
with st.expander("Filters", expanded=False):

    col1, col2 = st.columns(2)

    with col1:
        payment = st.multiselect(
            "Payment Type",
            options=sorted(df["payment_type"].unique()),
            default=sorted(df["payment_type"].unique())
        )

    with col2:
        risk = st.multiselect(
            "Risk Level",
            options=sorted(df["risk"].unique()),
            default=sorted(df["risk"].unique())
        )

# =====================================================
# APPLY FILTERS
# =====================================================
df = df[
    (df["payment_type"].isin(payment)) &
    (df["risk"].isin(risk))
]

if df.empty:
    st.warning("No data available for selected filters")
    st.stop()

# =====================================================
# KPI SECTION
# =====================================================
total_orders = len(df)
returned = len(df[df["status"] == "Returned"])
rto = (returned / total_orders) * 100
avg_delay = df["delay"].mean()

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Orders", total_orders)
k2.metric("Returned Orders", returned)
k3.metric("RTO %", f"{rto:.2f}%")
k4.metric("Average Delay", f"{avg_delay:.2f} days")

st.divider()

# =====================================================
# INSIGHTS
# =====================================================
st.subheader("Key Insights")

if rto > 50:
    st.error("High return rate detected")
elif rto > 30:
    st.warning("Moderate return rate detected")
else:
    st.success("Return rate is stable")

st.divider()

# =====================================================
# CHARTS
# =====================================================
c1, c2 = st.columns(2)

with c1:

    fig1 = px.histogram(
        df,
        x="risk",
        color="risk",
        title="Risk Distribution",
        color_discrete_map={
            "High": "#ef4444",
            "Medium": "#f59e0b",
            "Low": "#22c55e"
        }
    )

    fig1.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827")
    )

    st.plotly_chart(fig1, use_container_width=True)

with c2:

    fig2 = px.histogram(
        df,
        x="delay",
        title="Delay Distribution",
        color_discrete_sequence=["#2563eb"]
    )

    fig2.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#111827")
    )

    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =====================================================
# PAYMENT ANALYSIS
# =====================================================
st.subheader("Payment vs Returns")

fig3 = px.histogram(
    df,
    x="payment_type",
    color="status",
    barmode="group",
    color_discrete_map={
        "Returned": "#ef4444",
        "Delivered": "#22c55e"
    }
)

fig3.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#111827")
)

st.plotly_chart(fig3, use_container_width=True)

st.divider()

# =====================================================
# PREDICTION SECTION
# =====================================================
st.subheader("Prediction")

delay_input = st.slider(
    "Delivery Delay",
    1,
    15,
    5
)

if delay_input > 7:
    st.error("Prediction: High Return Risk")
elif delay_input > 4:
    st.warning("Prediction: Medium Return Risk")
else:
    st.success("Prediction: Low Return Risk")

st.divider()

# =====================================================
# TABLE
# =====================================================
st.subheader("High Risk Orders")

st.dataframe(
    df[df["risk"] == "High"].head(20),
    use_container_width=True
)