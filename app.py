import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="TintBox Analytics", layout="wide")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data.csv")

# -------------------------
# DATA CLEANING
# -------------------------
df.columns = df.columns.str.strip()

df['delay'] = pd.to_numeric(df.get('delay'), errors='coerce')
df['attempts'] = pd.to_numeric(df.get('attempts'), errors='coerce')

df['risk'] = df.get('risk').fillna("Unknown")
df['payment_type'] = df.get('payment_type').fillna("Unknown")
df['status'] = df.get('status').fillna("Unknown")

df = df.dropna(subset=['delay'])

# -------------------------
# MINIMAL PROFESSIONAL CSS
# -------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #f5f5f5;
}
h1 {
    font-weight: 600;
}
h2, h3 {
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("TintBox Order Intelligence Dashboard")
st.markdown("### Order Analytics and RTO Monitoring")

st.markdown("---")

# -------------------------
# SIDEBAR FILTERS
# -------------------------
st.sidebar.header("Filters")

payment_options = sorted(df['payment_type'].dropna().unique())
risk_options = sorted(df['risk'].dropna().unique())

payment = st.sidebar.multiselect(
    "Payment Type",
    options=payment_options,
    default=payment_options
)

risk = st.sidebar.multiselect(
    "Risk Level",
    options=risk_options,
    default=risk_options
)

if payment:
    df = df[df['payment_type'].isin(payment)]

if risk:
    df = df[df['risk'].isin(risk)]

if df.empty:
    st.warning("No data available for selected filters")
    st.stop()

# -------------------------
# KPI SECTION
# -------------------------
total_orders = len(df)
returned = len(df[df['status'] == 'Returned'])
rto = (returned / total_orders) * 100 if total_orders > 0 else 0
avg_delay = df['delay'].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Orders", total_orders)
col2.metric("Returned Orders", returned)
col3.metric("RTO Percentage", f"{rto:.2f}%")
col4.metric("Average Delay (days)", f"{avg_delay:.2f}")

st.markdown("---")

# -------------------------
# INSIGHTS
# -------------------------
st.subheader("Key Insights")

if rto > 50:
    st.error("High return rate detected. Immediate action recommended.")
elif rto > 30:
    st.warning("Moderate return rate. Monitor closely.")
else:
    st.success("Return rate is under control.")

st.info("Observation: Orders with high delay and cash payment show higher return probability.")

st.markdown("---")

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
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig1.update_layout(template="plotly")
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    fig2 = px.histogram(
        df,
        x="delay",
        title="Delay Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig2.update_layout(template="plotly")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# -------------------------
# PAYMENT ANALYSIS
# -------------------------
st.subheader("Payment Type vs Returns")

fig3 = px.histogram(
    df,
    x="payment_type",
    color="status",
    barmode="group",
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig3.update_layout(template="plotly")

st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# -------------------------
# HIGH RISK ORDERS
# -------------------------
st.subheader("High Risk Orders")

st.dataframe(df[df['risk'] == 'High'].head(20))

st.markdown("---")

# -------------------------
# ACTION DISTRIBUTION
# -------------------------
st.subheader("Recommended Actions Distribution")

fig4 = px.pie(
    df,
    names="action",
    title="Decision Engine Output",
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig4.update_layout(template="plotly")

st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# -------------------------
# RAW DATA
# -------------------------
with st.expander("View Full Dataset"):
    st.dataframe(df)