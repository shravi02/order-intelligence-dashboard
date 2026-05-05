import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="TintBox Analytics", layout="wide")

# -------------------------
# LOAD & CLEAN DATA
# -------------------------
df = pd.read_csv("data.csv")

# 🔧 DATA CLEANING
df = df.dropna()

df['delay'] = pd.to_numeric(df['delay'], errors='coerce')
df['attempts'] = pd.to_numeric(df['attempts'], errors='coerce')

df = df.dropna(subset=['delay'])

# -------------------------
# 🎨 UI STYLING
# -------------------------
st.markdown("""
<style>
.stApp { background-color: #f8fafc; }
section[data-testid="stSidebar"] { background-color: #e6f4ea; }

h1 { color: #1b5e20; }
h2, h3 { color: #2e7d32; }

[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #e0e0e0;
}

.plot-container { background-color: white !important; }

button {
    background-color: #2e7d32 !important;
    color: white !important;
}

[data-testid="stDataFrame"] { background-color: white; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("📦 TintBox Order Intelligence System")
st.caption("RTO Optimization | Final Year CSE Project")
st.markdown("### 📊 Real-time Order Analytics Dashboard")

st.divider()

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("🔍 Filters")

# Payment filter
payment_options = df['payment_type'].dropna().unique()
payment = st.sidebar.multiselect(
    "Payment Type",
    options=payment_options,
    default=payment_options
)

# Risk filter
risk_options = df['risk'].dropna().unique()
risk = st.sidebar.multiselect(
    "Risk Level",
    options=risk_options,
    default=risk_options
)

# Apply filters
df = df[
    (df['payment_type'].isin(payment)) &
    (df['risk'].isin(risk))
]

# 🚨 STOP EARLY IF NO DATA
if df.empty:
    st.warning("No data available for selected filters")
    st.stop()

# Optional: sort for better charts
df = df.sort_values(by="delay")

# -------------------------
# KPIs
# -------------------------
total_orders = len(df)
returned = len(df[df['status'] == 'Returned'])
rto = (returned / total_orders) * 100
avg_delay = df['delay'].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", total_orders)
col2.metric("Returned Orders", returned)
col3.metric("RTO %", f"{rto:.2f}%")
col4.metric("Avg Delay", f"{avg_delay:.2f} days")

st.divider()

# -------------------------
# INSIGHTS
# -------------------------
st.subheader("📌 Key Insights")

if rto > 50:
    st.error("⚠️ High RTO detected → Immediate intervention required")
elif rto > 30:
    st.warning("Moderate RTO → Monitor closely")
else:
    st.success("RTO under control")

st.info("💡 Insight: COD + High Delay = Highest RTO Risk")

st.divider()

# -------------------------
# CHARTS
# -------------------------
colA, colB = st.columns(2)

with colA:
    fig1 = px.bar(
        df,
        x="risk",
        color="risk",
        title="Risk Distribution",
        color_discrete_map={
            "High": "#e53935",
            "Medium": "#fb8c00",
            "Low": "#43a047"
        }
    )
    fig1.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    fig2 = px.histogram(
        df,
        x="delay",
        title="Delay Distribution",
        color_discrete_sequence=["#2e7d32"]
    )
    fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# -------------------------
# PAYMENT ANALYSIS
# -------------------------
st.subheader("💳 Payment vs RTO")

fig3 = px.histogram(
    df,
    x="payment_type",
    color="status",
    barmode="group",
    color_discrete_map={
        "Returned": "#e53935",
        "Delivered": "#43a047"
    }
)
fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# -------------------------
# HIGH RISK ORDERS
# -------------------------
st.subheader("🚨 High Risk Orders")
st.dataframe(df[df['risk'] == 'High'].head(20))

st.divider()

# -------------------------
# ACTION DISTRIBUTION
# -------------------------
st.subheader("🤖 Decision Engine Output")

fig4 = px.pie(
    df,
    names="action",
    title="Suggested Actions",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# -------------------------
# RAW DATA
# -------------------------
with st.expander("📋 View Full Dataset"):
    st.dataframe(df)