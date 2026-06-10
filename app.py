import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai

from risk_profiles import RISK_PROFILES
from asset_mapping import ASSET_MAPPING


model = joblib.load("models/investor_profile_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
features = joblib.load("models/features.pkl")


st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💰",
    layout="wide"
)


st.title("AI Financial Advisor")
st.write(
    "Machine learning powered investor profiling, portfolio allocation, projection, and Gemini AI explanation."
)


st.sidebar.header("User Financial Information")

age = st.sidebar.slider("Age", 18, 75, 34)
salary = st.sidebar.number_input("Annual Salary", min_value=10000, value=50000)
monthly_expenses = st.sidebar.number_input("Monthly Expenses", min_value=0, value=2500)
savings = st.sidebar.number_input("Current Savings", min_value=0, value=10000)
debt = st.sidebar.number_input("Total Debt", min_value=0, value=5000)
dependents = st.sidebar.slider("Number of Dependents", 0, 5, 0)
investment_horizon = st.sidebar.slider("Investment Horizon Years", 1, 40, 20)
risk_score = st.sidebar.slider("Risk Score", 1, 10, 7)

st.sidebar.header("Projection Inputs")

monthly_contribution = st.sidebar.number_input(
    "Monthly Investment Contribution",
    min_value=0,
    value=300
)

expected_return = st.sidebar.slider(
    "Expected Annual Return (%)",
    1.0,
    15.0,
    7.0
)


monthly_income = salary / 12

savings_rate = (
    (monthly_income - monthly_expenses) / monthly_income
    if monthly_income > 0
    else 0
)

emergency_fund_months = (
    savings / monthly_expenses
    if monthly_expenses > 0
    else 0
)

debt_to_income = (
    debt / salary
    if salary > 0
    else 0
)


user_data = {
    "age": age,
    "salary": salary,
    "monthly_expenses": monthly_expenses,
    "savings": savings,
    "debt": debt,
    "dependents": dependents,
    "investment_horizon": investment_horizon,
    "risk_score": risk_score,
    "savings_rate": savings_rate,
    "emergency_fund_months": emergency_fund_months,
    "debt_to_income": debt_to_income
}

input_df = pd.DataFrame([user_data])
input_df = input_df[features]


def calculate_risk_score_100(profile, risk_score, investment_horizon, savings_rate, debt_to_income):
    base_scores = {
        "Conservative": 30,
        "Balanced": 55,
        "Aggressive": 80
    }

    score = base_scores[profile]
    score += (risk_score - 5) * 2
    score += min(investment_horizon, 30) * 0.3
    score += savings_rate * 10
    score -= debt_to_income * 15

    return max(0, min(100, round(score, 1)))


def calculate_future_value(monthly_contribution, annual_return, years):
    monthly_rate = annual_return / 100 / 12
    months = years * 12

    if monthly_rate == 0:
        return monthly_contribution * months

    fv = monthly_contribution * (((1 + monthly_rate) ** months - 1) / monthly_rate)

    return round(fv, 2)


def generate_simple_explanation(user_data):
    factors = []

    if user_data["investment_horizon"] >= 20:
        factors.append("Long investment horizon supports higher growth-oriented allocation.")
    elif user_data["investment_horizon"] < 5:
        factors.append("Short investment horizon reduces suitability for aggressive allocation.")

    if user_data["savings_rate"] >= 0.30:
        factors.append("Strong savings rate improves financial flexibility.")
    elif user_data["savings_rate"] < 0.10:
        factors.append("Low savings rate suggests a more cautious approach.")

    if user_data["emergency_fund_months"] >= 6:
        factors.append("Strong emergency fund supports investment risk-taking.")
    elif user_data["emergency_fund_months"] < 3:
        factors.append("Weak emergency fund suggests prioritising cash reserves.")

    if user_data["debt_to_income"] > 0.50:
        factors.append("High debt-to-income ratio reduces risk capacity.")
    else:
        factors.append("Manageable debt-to-income ratio supports investment flexibility.")

    if user_data["risk_score"] >= 8:
        factors.append("High stated risk tolerance supports aggressive allocation.")
    elif user_data["risk_score"] <= 3:
        factors.append("Low stated risk tolerance supports conservative allocation.")

    return factors


def generate_gemini_explanation(profile, allocation_df, user_data, risk_score_100, projected_value):
    try:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return "Gemini explanation unavailable. Please set your GEMINI_API_KEY environment variable."

        genai.configure(api_key=api_key)

        model_gemini = genai.GenerativeModel("gemini-3.5-flash")

        prompt = f"""
        You are explaining an educational AI financial advisor demo.

        Important rules:
        - Do not provide regulated financial advice.
        - Do not tell the user to buy specific products.
        - Keep the explanation concise.
        - Use simple language.
        - Mention that this is for educational purposes only.

        Investor profile: {profile}
        Risk score: {risk_score_100}/100

        User data:
        {user_data}

        Portfolio allocation:
        {allocation_df.to_string(index=False)}

        Projected portfolio value:
        £{projected_value:,.0f}

        Explain:
        1. Why this profile was selected
        2. What the allocation means
        3. What the user should be careful about
        4. Educational disclaimer
        """

        response = model_gemini.generate_content(prompt)

        if hasattr(response, "text"):
            return response.text

        return "Gemini did not return a text response."

    except Exception as e:
        return f"Gemini Error: {str(e)}"


if st.button("Generate Recommendation"):

    prediction_encoded = model.predict(input_df)[0]
    profile = label_encoder.inverse_transform([prediction_encoded])[0]

    allocation = RISK_PROFILES[profile]

    risk_score_100 = calculate_risk_score_100(
        profile,
        risk_score,
        investment_horizon,
        savings_rate,
        debt_to_income
    )

    projected_value = calculate_future_value(
        monthly_contribution,
        expected_return,
        investment_horizon
    )

    allocation_df = pd.DataFrame({
        "Asset Class": list(allocation.keys()),
        "Allocation (%)": list(allocation.values())
    })

    allocation_df["Example Instrument"] = allocation_df["Asset Class"].map(ASSET_MAPPING)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Dashboard",
            "Portfolio",
            "AI Explanation",
            "Disclaimer"
        ]
    )

    with tab1:
        st.subheader("Investor Profile")
        st.success(profile)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Savings Rate", f"{savings_rate:.2%}")
            st.metric("Emergency Fund", f"{emergency_fund_months:.1f} months")

        with col2:
            st.metric("Debt-to-Income", f"{debt_to_income:.2%}")
            st.metric("Projected Value", f"£{projected_value:,.0f}")

        st.subheader("Risk Score")

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_score_100,
                title={"text": "Risk Score / 100"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "green"}
                }
            )
        )

        gauge.update_layout(height=350)

        st.plotly_chart(gauge, use_container_width=True)

        st.subheader("Why This Recommendation?")

        explanation_factors = generate_simple_explanation(user_data)

        for factor in explanation_factors:
            st.write(f"- {factor}")

    with tab2:
        st.subheader("Portfolio Allocation")

        st.dataframe(allocation_df, use_container_width=True)

        fig = px.pie(
            allocation_df,
            names="Asset Class",
            values="Allocation (%)",
            title=f"{profile} Portfolio Allocation"
        )

        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Portfolio Projection")

        projection_df = pd.DataFrame({
            "Metric": [
                "Monthly Contribution",
                "Expected Annual Return",
                "Investment Horizon",
                "Projected Value"
            ],
            "Value": [
                f"£{monthly_contribution:,.0f}",
                f"{expected_return:.1f}%",
                f"{investment_horizon} years",
                f"£{projected_value:,.0f}"
            ]
        })

        st.dataframe(projection_df, use_container_width=True)

    with tab3:
        st.subheader("Gemini AI Explanation")

        with st.spinner("Generating Gemini explanation..."):
            gemini_explanation = generate_gemini_explanation(
                profile,
                allocation_df,
                user_data,
                risk_score_100,
                projected_value
            )

        st.write(gemini_explanation)

    with tab4:
        st.subheader("Educational Disclaimer")

        st.warning(
            "This app is for educational and demonstration purposes only. "
            "It does not provide regulated financial advice, investment advice, "
            "or product recommendations."
        )

        st.write(
            """
            The portfolio allocations shown are hypothetical and based on a machine learning
            classification model trained on synthetic data.

            The example instruments are included only to demonstrate how asset classes may be
            represented in a real-world application. They are not recommendations to buy, sell,
            or hold any financial product.

            Past performance does not guarantee future results. Actual investment outcomes may
            differ significantly from projections.
            """
        )

else:
    st.info("Enter your financial information in the sidebar and click Generate Recommendation.")