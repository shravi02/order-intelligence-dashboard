import streamlit as st
import pandas as pd
import plotly.express as px
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Logistics Intelligence System",
    layout="wide"
)

# ---------------- LOAD LOGO ----------------
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64("logo.png")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data.csv")

df = df.dropna()

df["delay"] = pd.to_numeric(df["delay"], errors="coerce")
df["attempts"] = pd.to_numeric(df["attempts"], errors="coerce")

df = df.dropna(subset=["delay"])

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main App */
.stApp {
    background-color: #f8fafc;
}

/* Remove extra top spacing */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

/* Header */
.header-container {
    text-align: center;
    margin-top: 10px;
    margin-bottom: 20px;
}

/* Logo */
.logo-img {
    width: 220px;
    margin-bottom: 10px;
}

/* Title */
.title-text {
    font-size: 38px;
    font-weight: 700;
    color: #1e293b;
    margin-top: 10px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 15px;
    border: 1px solid #e5e7eb;
}

/* Charts */
.plot-container {
    border-radius: 12px;
    overflow: hidden;
}

/* Mobile Responsive */
@media (max-width: 768px) {

    .logo-img {
        width: 160px;
    }

    .title-text {
        font-size: 28px;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    f"""
    <div class="header-container">

        <img class="logo-img"
        src="data:image/png;base64,{logo_base64}">

        <div class="title-text">
            Logistics Intelligence System
        </div>

    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- FILTERS ----------------
with st.expander("Filters", expanded=False):

    col1, col2 = st.columns(2)

    with col1:
        payment = st.multiselect(
            "Payment Type",
            options=df["payment_type"].unique(),
            default=df["payment_type"].unique()
        )

    with col2:
        risk = st.multiselect(
            "Risk Level",
            options=df["risk"].unique(),
            default=df["risk"].unique()
        )

# ---------------- APPLY FILTERS ----------------
df = df[
    (df["payment_type"].isin(payment)) &
    (df["risk"].isin(risk))
]

# ---------------- KPIs ----------------
total_orders = len(df)
returned = len(df[df["status"] == "Returned"])

rto = (returned / total_orders) * 100 if total_orders > 0 else 0

avg_delay = df["delay"].mean()

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Orders", total_orders)
k2.metric("Returned Orders", returned)
k3.metric("RTO %", f"{rto:.2f}%")
k4.metric("Average Delay", f"{avg_delay:.2f} days")

st.divider()

# ---------------- INSIGHTS ----------------
st.subheader("Key Insights")

if rto > 50:
    st.error("High return rate detected")

elif rto > 30:
    st.warning("Moderate return rate detected")

else:
    st.success("Return rate is stable")

st.divider()

# ---------------- CHARTS ----------------
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
        plot_bgcolor="white"
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
        plot_bgcolor="white"
    )

    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------- PAYMENT ANALYSIS ----------------
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
    plot_bgcolor="white"
)

st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------- ML PREDICTION ----------------
st.subheader("ML Prediction")

delay_input = st.slider(
    "Delivery Delay (Days)",
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

# ---------------- DOWNLOAD REPORT ----------------
st.subheader("Download Reports")

csv = df.to_csv(index=False)

st.download_button(
    label="Download CSV Report",
    data=csv,
    file_name="logistics_report.csv",
    mime="text/csv"
)

st.divider()

# ---------------- UPLOAD DATASET ----------------
st.subheader("Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully")

    st.dataframe(
        uploaded_df.head(),
        use_container_width=True
    )

st.divider()

# ---------------- HIGH RISK ORDERS ----------------
st.subheader("High Risk Orders")

st.dataframe(
    df[df["risk"] == "High"].head(20),
    use_container_width=True
)