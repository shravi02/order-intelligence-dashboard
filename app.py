import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# PAGE CONFIG (MUST BE FIRST)
# -------------------------
st.set_page_config(page_title="TintBox Analytics", layout="wide")

st.write("VERSION: FINAL FIX V5")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data.csv")

# -------------------------
# 🔧 SAFE DATA CLEANING (FIXED)
# -------------------------
df.columns = df.columns.str.strip()

# Convert numeric columns safely
df['delay'] = pd.to_numeric(df.get('delay'), errors='coerce')
df['attempts'] = pd.to_numeric(df.get('attempts'), errors='coerce')

# Fill missing categorical values (IMPORTANT)
df['risk'] = df.get('risk').fillna("Unknown")
df['payment_type'] = df.get('payment_type').fillna("Unknown")
df['status'] = df.get('status').fillna("Unknown")

# Remove only rows where delay is invalid
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

payment = st.sidebar.multiselect(
    "Payment Type",
    options=df['payment_type'].unique(),
    default=df['payment_type'].unique()
)

risk = st.sidebar.multiselect(
    "Risk Level",
    options=df['risk'].unique(),
    default=df['risk'].unique()
)

df = df[
    (df['payment_type'].isin(payment)) &
    (df['risk'].isin(risk))
]

if df.empty:
    st.warning("No data available for selected filters")
    st.stop()

# -------------------------
# KPIs
# -------------------------
total_orders = len(df)
returned = len(df[df['status'] == 'Returned'])
rto = (returned / total_orders) * 100 if total_orders > 0 else 0
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
# CHARTS (SAFE VERSION)
# -------------------------
colA, colB = st.columns(2)

with colA:
    try:
        fig1 = px.bar(df, x="risk", color="risk", title="Risk Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

with colB:
    try:
        fig2 = px.histogram(df, x="delay", title="Delay Distribution")
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

st.divider()

# -------------------------
# PAYMENT ANALYSIS
# -------------------------
st.subheader("💳 Payment vs RTO")

try:
    fig3 = px.histogram(df, x="payment_type", color="status", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)
except Exception as e:
    st.error(f"Chart error: {e}")

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

try:
    fig4 = px.pie(df, names="action", title="Suggested Actions")
    st.plotly_chart(fig4, use_container_width=True)
except Exception as e:
    st.error(f"Chart error: {e}")

st.divider()

# -------------------------
# RAW DATA
# -------------------------
with st.expander("📋 View Full Dataset"):
    st.dataframe(df)
