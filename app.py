import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import base64

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Smart RTO & Delivery Analytics",
    layout="wide"
)

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
# LOAD LOGO (HIGH QUALITY)
# -------------------------
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    logo_base64 = get_base64("logo.png")
except:
    logo_base64 = ""

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown("""
<style>

/* Main container */
.block-container {
    padding-top: 1.5rem;
}

/* Header styling */
.header-container {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 10px;
}

/* Logo */
.logo-img {
    height: 90px;
    width: auto;
    object-fit: contain;
}

/* Title */
.title-text {
    font-size: 24px;
    font-weight: 600;
    margin: 0;
    display: flex;
    align-items: center;
    height: 90px;
}

/* KPI Cards */
[data-testid="metric-container"] {
    border: 1px solid #e6e6e6;
    padding: 15px;
    border-radius: 12px;
    background-color: #ffffff;
}

/* Mobile Responsive */
@media (max-width: 768px) {

    .header-container {
        flex-direction: column;
        align-items: center;
        text-align: center;
    }

    .logo-img {
        height: 75px;
    }

    .title-text {
        font-size: 20px;
        height: auto;
    }
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.markdown(f"""
<div class="header-container">
    <img src="data:image/png;base64,{logo_base64}" class="logo-img">
    <div class="title-text">
        Logistics Intelligence System
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# FILTERS
# -------------------------
payment_options = sorted(df['payment_type'].unique())
risk_options = sorted(df['risk'].unique())

with st.expander("Filters", expanded=False):

    payment = st.multiselect(
        "Payment Type",
        payment_options,
        default=payment_options
    )

    risk = st.multiselect(
        "Risk Level",
        risk_options,
        default=risk_options
    )

# Apply filters
if payment:
    df = df[df['payment_type'].isin(payment)]

if risk:
    df = df[df['risk'].isin(risk)]

if df.empty:
    st.warning("No data available for selected filters")
    st.stop()

# -------------------------
# TABS
# -------------------------
tab1, tab2, tab3 = st.tabs([
    "Dashboard",
    "Data Explorer",
    "Prediction"
])

# ====================================================
# DASHBOARD TAB
# ====================================================
with tab1:

    total_orders = len(df)
    returned = len(df[df['status'] == 'Returned'])
    rto = (returned / total_orders) * 100 if total_orders > 0 else 0
    avg_delay = df['delay'].mean()

    # KPI ROW
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Orders", total_orders)
    col2.metric("Returned Orders", returned)
    col3.metric("RTO %", f"{rto:.2f}%")
    col4.metric("Average Delay", f"{avg_delay:.2f} days")

    st.markdown("---")

    # INSIGHTS
    st.subheader("Key Insights")

    if rto > 50:
        st.error("High return rate detected")
    elif rto > 30:
        st.warning("Moderate return rate")
    else:
        st.success("Return rate under control")

    st.markdown("---")

    # CHARTS
    colA, colB = st.columns(2)

    with colA:
        fig1 = px.bar(
            df,
            x="risk",
            color="risk",
            title="Risk Distribution"
        )

        fig1.update_layout(template="plotly")
        st.plotly_chart(fig1, use_container_width=True)

    with colB:
        fig2 = px.histogram(
            df,
            x="delay",
            title="Delay Distribution"
        )

        fig2.update_layout(template="plotly")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # PAYMENT ANALYSIS
    fig3 = px.histogram(
        df,
        x="payment_type",
        color="status",
        barmode="group",
        title="Payment vs Returns"
    )

    fig3.update_layout(template="plotly")

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # HIGH RISK ORDERS
    st.subheader("High Risk Orders")

    st.dataframe(
        df[df['risk'] == 'High'].head(20),
        use_container_width=True
    )

# ====================================================
# DATA EXPLORER TAB
# ====================================================
with tab2:

    st.subheader("Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

# ====================================================
# ML PREDICTION TAB
# ====================================================
with tab3:

    st.subheader("Return Prediction Model")

    # Target column
    df['target'] = df['status'].apply(
        lambda x: 1 if x == "Returned" else 0
    )

    # Features
    X = df[['delay', 'attempts']]
    y = df['target']

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Inputs
    delay_input = st.slider(
        "Delay (days)",
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

    # Prediction
    if st.button("Predict Return Risk"):

        prediction = model.predict(
            [[delay_input, attempts_input]]
        )[0]

        if prediction == 1:
            st.error("High probability of return")
        else:
            st.success("Likely to be delivered")