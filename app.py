import streamlit as st
import streamlit_authenticator as stauth
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.platypus import Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Logistics Intelligence System",
    layout="wide"
)

# =====================================================
# AUTHENTICATION
# =====================================================
names = ["Admin User", "Viewer User"]
usernames = ["admin", "viewer"]
credentials = {
    "usernames": {
        "admin": {
            "name": "Admin User",
            "password": "admin123"
        },
        "viewer": {
            "name": "Viewer User",
            "password": "viewer123"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "dashboard_cookie",
    "abcdef",
    cookie_expiry_days=1
)

st.info("""
Admin Login

Username: admin
Password: admin123
""")
authenticator.login(location="main")

name = st.session_state.get("name")
authentication_status = st.session_state.get("authentication_status")
username = st.session_state.get("username")

if authentication_status == False:
    st.error("Incorrect username or password")

elif authentication_status == None:
    st.warning("Please login to continue")

elif authentication_status:

    # =================================================
    # LOAD DATA
    # =================================================
    DEFAULT_FILE = "data.csv"

    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(DEFAULT_FILE)

    # =================================================
    # DATA CLEANING
    # =================================================
    df.columns = df.columns.str.strip()

    numeric_cols = ["delay", "attempts", "order_value"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["delay"])

    for col in ["risk", "payment_type", "status", "city", "product"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # =================================================
    # CUSTOM CSS
    # =================================================
    st.markdown("""
    <style>
    .stApp { background-color: #f8fafc; }
    .block-container { padding-top: 1.5rem; padding-left: 2rem; padding-right: 2rem; }
    .main-title { text-align: center; font-size: 42px; font-weight: 700; color: #0f172a; margin-bottom: 20px; }
    [data-testid="metric-container"] { background: white; border-radius: 16px; padding: 20px; border: 1px solid #e2e8f0; }
    .stPlotlyChart { background: white; border-radius: 16px; padding: 10px; }
    .insight-box { background: white; padding: 15px; border-radius: 14px; border-left: 6px solid #2563eb; margin-bottom: 12px; font-size: 16px; }
    @media (max-width: 768px) { .main-title { font-size: 28px; } .block-container { padding-left: 1rem; padding-right: 1rem; } }
    </style>
    """, unsafe_allow_html=True)

    # =================================================
    # HEADER
    # =================================================
    st.markdown('''<div class="main-title">Logistics Intelligence System</div>''', unsafe_allow_html=True)
    st.markdown("---")

    # =================================================
    # SIDEBAR NAVIGATION
    # =================================================

    st.sidebar.image(
        "logo.png",
        width=180
    )

    st.sidebar.markdown(
        "## Logistics Intelligence"
    )

    st.sidebar.caption(
        "AI-Powered Business Analytics"
    )

    st.sidebar.markdown("---")

    authenticator.logout(
        "Logout",
        "sidebar"
    )

    st.sidebar.markdown("---")

    if username == "admin":

        section = st.sidebar.radio(
            "Navigation",
            [
                "Dashboard",
                "Business Health",
                "ML Prediction",
                "Forecasting",
                "Download Reports",
                "Upload Dataset"
            ]
        )

    else:

        section = st.sidebar.radio(
            "Navigation",
            [
                "Dashboard"
            ]
        )

    st.markdown("---")

    st.caption(

        "Logistics Intelligence System • AI-Powered Operational Analytics Platform"
    )
    # =================================================
    # FILTERS
    # =================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    payment = st.sidebar.multiselect(
        "Payment Type",
        options=sorted(df["payment_type"].unique()),
        default=sorted(df["payment_type"].unique())
    )

    risk = st.sidebar.multiselect(
        "Risk Level",
        options=sorted(df["risk"].unique()),
        default=sorted(df["risk"].unique())
    )

    filtered_df = df[
        (df["payment_type"].isin(payment)) &
        (df["risk"].isin(risk))
    ]

    if filtered_df.empty:
        st.warning("No data for selected filters.")
        st.stop()

    # =================================================
    # DASHBOARD
    # =================================================
    if section == "Dashboard":

        st.header("Executive Summary")

        total_orders = len(filtered_df)
        returned_orders = len(filtered_df[filtered_df["status"] == "Returned"])
        revenue = filtered_df["order_value"].sum()
        revenue_loss = filtered_df[filtered_df["status"] == "Returned"]["order_value"].sum()
        rto = (returned_orders / total_orders) * 100 if total_orders > 0 else 0
        high_risk_orders = len(filtered_df[filtered_df["risk"] == "High"])
        avg_delay = filtered_df["delay"].mean()

        k1, k2, k3 = st.columns(3)
        k4, k5, k6 = st.columns(3)

        k1.metric("Total Orders",      f"{total_orders:,}")
        k2.metric("Revenue",           f"\u20b9{revenue:,.0f}")
        k3.metric("Returned Orders",   f"{returned_orders:,}")
        k4.metric("RTO %",             f"{rto:.2f}%")
        k5.metric("Revenue Loss",      f"\u20b9{revenue_loss:,.0f}")
        k6.metric("High Risk Orders",  f"{high_risk_orders:,}")

        st.markdown("---")

        # Smart Insights
        st.subheader("Smart Insights")
        st.markdown('''<div class="insight-box">\u26a0 COD orders show highest return probability</div>''', unsafe_allow_html=True)
        st.markdown('''<div class="insight-box">\u26a0 Orders delayed beyond 7 days are most likely to return</div>''', unsafe_allow_html=True)
        st.markdown('''<div class="insight-box">\u2705 UPI orders have lowest RTO percentage</div>''', unsafe_allow_html=True)

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            city_revenue = filtered_df.groupby("city")["order_value"].sum().reset_index()
            fig1 = px.bar(city_revenue, x="city", y="order_value", title="Revenue by City", color="order_value")
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            product_returns = filtered_df[filtered_df["status"] == "Returned"]
            product_returns = product_returns.groupby("product").size().reset_index(name="returns")
            fig2 = px.bar(product_returns, x="product", y="returns", title="Top Returning Products", color="returns")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        st.subheader("Action Center")
        high_risk_df = filtered_df[filtered_df["risk"] == "High"].sort_values(by="delay", ascending=False)
        st.dataframe(
            high_risk_df[["order_id", "city", "product", "delay", "status", "action"]].head(20),
            use_container_width=True
        )

    # =================================================
    # BUSINESS HEALTH
    # =================================================
    elif section == "Business Health":

        st.header("Business Health Analytics")

        c1, c2 = st.columns(2)

        with c1:
            fig3 = px.histogram(filtered_df, x="payment_type", color="status", barmode="group", title="Payment Method Performance")
            fig3.update_layout(paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig3, use_container_width=True)

        with c2:
            fig4 = px.box(filtered_df, x="risk", y="delay", color="risk", title="Delay Distribution by Risk")
            fig4.update_layout(paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

        monthly_orders = filtered_df.groupby("order_date").size().reset_index(name="orders")
        fig5 = px.line(monthly_orders, x="order_date", y="orders", title="Order Trend Over Time")
        fig5.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("---")

        st.subheader("Geographic Analytics")

        city_rto = filtered_df.groupby("city").apply(
            lambda x: (x["status"] == "Returned").mean() * 100
        ).reset_index(name="RTO_Percentage")

        city_orders = filtered_df.groupby("city").size().reset_index(name="Total_Orders")
        city_revenue = filtered_df.groupby("city")["order_value"].sum().reset_index(name="Revenue")

        city_summary = pd.merge(city_rto, city_orders, on="city")
        city_summary = pd.merge(city_summary, city_revenue, on="city")

        fig6 = px.scatter(
            city_summary, x="city", y="RTO_Percentage",
            size="Total_Orders", color="Revenue",
            hover_name="city", title="City-wise RTO Risk Analysis", size_max=60
        )
        fig6.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown("---")

        st.subheader("Top Risky Cities")
        risky_cities = city_summary.sort_values(by="RTO_Percentage", ascending=False)
        st.dataframe(risky_cities, use_container_width=True)

        st.markdown("---")

        st.subheader("Top Returning Products")
        product_returns = filtered_df[filtered_df["status"] == "Returned"]
        product_returns = product_returns.groupby("product").size().reset_index(name="Returns")
        fig7 = px.bar(product_returns, x="product", y="Returns", color="Returns", title="Product-wise Return Volume")
        fig7.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig7, use_container_width=True)

    # =================================================
    # ML PREDICTION
    # =================================================
    elif section == "ML Prediction":

        st.header("AI Return Risk Prediction")

        model_df = filtered_df.copy()
        model_df["target"] = model_df["status"].apply(lambda x: 1 if x == "Returned" else 0)

        model_df = pd.get_dummies(model_df, columns=["payment_type", "city"])

        feature_columns = [
            col for col in model_df.columns
            if col not in ["status", "target", "order_id", "customer_name", "product", "action", "risk", "order_date", "product_type"]
        ]

        X = model_df[feature_columns]
        y = model_df["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model  = RandomForestClassifier()
        lr_model  = LogisticRegression(max_iter=1000)
        xgb_model = XGBClassifier(eval_metric="logloss", verbosity=0)

        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)

        rf_pred  = rf_model.predict(X_test)
        lr_pred  = lr_model.predict(X_test)
        xgb_pred = xgb_model.predict(X_test)

        rf_acc  = accuracy_score(y_test, rf_pred)
        lr_acc  = accuracy_score(y_test, lr_pred)
        xgb_acc = accuracy_score(y_test, xgb_pred)

        st.subheader("Model Comparison")
        comparison_df = pd.DataFrame({
            "Model":    ["Random Forest", "Logistic Regression", "XGBoost"],
            "Accuracy": [rf_acc, lr_acc, xgb_acc]
        })
        st.dataframe(comparison_df, use_container_width=True)

        st.markdown("---")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, rf_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown("---")

        st.subheader("ROC Curve")
        rf_probs = rf_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, rf_probs)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0, 1], [0, 1], linestyle="--")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("---")

        st.subheader("AI Risk Simulator")

        delay_input       = st.slider("Delivery Delay",   0,   15,   5)
        attempts_input    = st.slider("Delivery Attempts", 1,    5,   2)
        order_value_input = st.slider("Order Value",     200, 5000, 1500)

        if st.button("Predict Return Risk"):
            input_data = X.iloc[0:1].copy()
            input_data[:] = 0
            input_data["delay"]       = delay_input
            input_data["attempts"]    = attempts_input
            input_data["order_value"] = order_value_input
            result = rf_model.predict(input_data)[0]
            if result == 1:
                st.error("High Return Risk Predicted")
                st.warning("Recommended Action: Call Customer")
            else:
                st.success("Low Return Risk Predicted")
                st.info("Recommended Action: Normal Delivery")

    # =================================================
    # FORECASTING
    # =================================================
    elif section == "Forecasting":

        st.header("Return Order Forecasting")

        forecast_df = filtered_df.copy()
        forecast_df["order_date"] = pd.to_datetime(forecast_df["order_date"], dayfirst=True, errors="coerce")

        returns_df = forecast_df[forecast_df["status"] == "Returned"]
        daily_returns = returns_df.groupby("order_date").size().reset_index(name="y")
        daily_returns.columns = ["ds", "y"]

        if len(daily_returns) < 2:
            st.warning("Not enough return data to forecast. Try removing filters.")
        else:
            model = Prophet()
            model.fit(daily_returns)
            future   = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)

            st.subheader("Next 30 Days Return Forecast")
            fig1 = px.line(forecast, x="ds", y="yhat", title="Predicted Return Orders")
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown("---")

            st.subheader("Forecast Confidence Interval")
            fig2 = px.line(forecast, x="ds", y="yhat_upper", title="Forecast with Confidence Band")
            fig2.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Forecast")
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")

            avg_forecast = forecast["yhat"].tail(30).mean()
            if avg_forecast > 20:
                st.error("High future return trend predicted")
            elif avg_forecast > 10:
                st.warning("Moderate return growth predicted")
            else:
                st.success("Stable return trend predicted")

    # =================================================
    # DOWNLOAD REPORTS
    # =================================================
    elif section == "Download Reports":

        st.header("Download Reports")

        csv = filtered_df.to_csv(index=False)
        st.download_button(label="Download Full Analytics CSV", data=csv, file_name="analytics_report.csv", mime="text/csv")

        st.markdown("---")

        high_risk_csv = filtered_df[filtered_df["risk"] == "High"].to_csv(index=False)
        st.download_button(label="Download High Risk Orders CSV", data=high_risk_csv, file_name="high_risk_orders.csv", mime="text/csv")

        st.markdown("---")

        st.subheader("Executive PDF Report")

        total_orders    = len(filtered_df)
        returned_orders = len(filtered_df[filtered_df["status"] == "Returned"])
        revenue         = filtered_df["order_value"].sum()
        revenue_loss    = filtered_df[filtered_df["status"] == "Returned"]["order_value"].sum()
        rto             = (returned_orders / total_orders) * 100 if total_orders > 0 else 0
        high_risk_orders = len(filtered_df[filtered_df["risk"] == "High"])

        buffer = BytesIO()
        doc    = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Logistics Intelligence Executive Report", styles["Title"]))
        elements.append(Spacer(1, 20))

        summary_text = (
            f"<b>Total Orders:</b> {total_orders}<br/><br/>"
            f"<b>Total Revenue:</b> Rs.{revenue:,.0f}<br/><br/>"
            f"<b>Returned Orders:</b> {returned_orders}<br/><br/>"
            f"<b>RTO Percentage:</b> {rto:.2f}%<br/><br/>"
            f"<b>Revenue Loss:</b> Rs.{revenue_loss:,.0f}<br/><br/>"
            f"<b>High Risk Orders:</b> {high_risk_orders}<br/><br/>"
        )
        elements.append(Paragraph(summary_text, styles["BodyText"]))
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("Business Insights", styles["Heading2"]))
        insights_text = (
            "- COD orders show highest return probability<br/><br/>"
            "- Orders delayed beyond 7 days have significantly higher RTO<br/><br/>"
            "- UPI orders show strongest delivery success rate<br/><br/>"
            "- High-risk orders require immediate operational action<br/><br/>"
        )
        elements.append(Paragraph(insights_text, styles["BodyText"]))
        elements.append(Spacer(1, 20))

        top_cities = filtered_df.groupby("city").apply(
            lambda x: (x["status"] == "Returned").mean() * 100
        ).reset_index(name="RTO")
        top_cities = top_cities.sort_values(by="RTO", ascending=False).head(5)

        city_text = "<b>Top Risky Cities:</b><br/><br/>"
        for _, row in top_cities.iterrows():
            city_text += f"{row['city']} - {row['RTO']:.2f}% RTO<br/>"

        elements.append(Paragraph(city_text, styles["BodyText"]))

        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()

        st.download_button(
            label="Generate Executive PDF Report",
            data=pdf,
            file_name="executive_report.pdf",
            mime="application/pdf"
        )

    # =================================================
    # UPLOAD DATASET
    # =================================================
    elif section == "Upload Dataset":

        st.header("Upload Dataset")
        st.write("Upload a CSV dataset for analytics.")

        if uploaded_file is not None:
            st.success("Dataset uploaded successfully")
            st.dataframe(df.head(), use_container_width=True)
            st.write("Rows:", df.shape[0])
            st.write("Columns:", df.shape[1])
        else:
            st.info("Currently using default dataset")