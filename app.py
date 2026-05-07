import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import base64

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

numeric_cols = ['delay', 'attempts']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'delay' in df.columns:
    df = df.dropna(subset=['delay'])

text_cols = ['risk', 'payment_type', 'status']

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown')

# =====================================================
# LOAD LOGO
# =====================================================
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    logo_base64 = get_base64("logo.png")
except:
    logo_base64 = ""

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

/* Main App */
.stApp {
    background-color: #f8fafc;
}

/* Page Padding */
.block-container {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Metric Cards */
[data-testid="metric-container"] {
    background: white;
    border-radius: 14px;
    padding: 18px;
    border: 1px solid #e5e7eb;
}

/* Charts */
.stPlotlyChart {
    background: white;
    border-radius: 14px;
    padding: 10px;
}

/* Mobile Responsive */
@media (max-width: 768px) {

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
st.markdown(f"""
<div style='text-align:center; margin-bottom:20px;'>

    <img 
        src='data:image/png;base64,{logo_base64}'
        width='120'
    >

    <h1 style='
        margin-top:10px;
        font-size:38px;
        font-weight:700;
        color:#1e293b;
    '>
        Logistics Intelligence System
    </h1>

</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
section = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Advanced Analytics",
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

# Apply filters
filtered_df = df[
    (df['payment_type'].isin(payment)) &
    (df['risk'].isin(risk))
]

if filtered_df.empty:
    st.warning("No data available for selected filters")
    st.stop()

# =====================================================
# DASHBOARD
# =====================================================
if section == "Dashboard":

    st.header("Dashboard")

    total_orders = len(filtered_df)
    returned = len(filtered_df[filtered_df['status'] == 'Returned'])

    rto = (
        (returned / total_orders) * 100
        if total_orders > 0 else 0
    )

    avg_delay = filtered_df['delay'].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Orders", total_orders)
    col2.metric("Returned Orders", returned)
    col3.metric("RTO %", f"{rto:.2f}%")
    col4.metric("Average Delay", f"{avg_delay:.2f} days")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:

        fig1 = px.histogram(
            filtered_df,
            x='risk',
            color='risk',
            title='Risk Distribution'
        )

        st.plotly_chart(fig1, use_container_width=True)

    with c2:

        fig2 = px.histogram(
            filtered_df,
            x='delay',
            title='Delay Distribution'
        )

        st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# ADVANCED ANALYTICS
# =====================================================
elif section == "Advanced Analytics":

    st.header("Advanced Analytics")

    fig3 = px.scatter(
        filtered_df,
        x='delay',
        y='attempts',
        color='risk',
        title='Delay vs Attempts'
    )

    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.histogram(
        filtered_df,
        x='payment_type',
        color='status',
        barmode='group',
        title='Payment vs Returns'
    )

    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.box(
        filtered_df,
        x='risk',
        y='delay',
        color='risk',
        title='Delay Distribution by Risk'
    )

    st.plotly_chart(fig5, use_container_width=True)

# =====================================================
# ML PREDICTION
# =====================================================
elif section == "ML Prediction":

    st.header("ML Prediction")

    model_df = filtered_df.copy()

    model_df['target'] = model_df['status'].apply(
        lambda x: 1 if x == 'Returned' else 0
    )

    X = model_df[['delay', 'attempts']]
    y = model_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    st.metric(
        "Model Accuracy",
        f"{accuracy*100:.2f}%"
    )

    st.markdown("---")

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

    if st.button("Predict Return Risk"):

        result = model.predict(
            [[delay_input, attempts_input]]
        )[0]

        if result == 1:
            st.error("High Return Risk")
        else:
            st.success("Low Return Risk")

# =====================================================
# DOWNLOAD REPORTS
# =====================================================
elif section == "Download Reports":

    st.header("Download Reports")

    csv = filtered_df.to_csv(index=False)

    st.download_button(
        label="Download Filtered CSV",
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
        "Upload a CSV file to analyze logistics and return performance."
    )

    if uploaded_file is not None:

        st.success("Dataset uploaded successfully")

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

    else:

        st.info("Currently using default dataset")