import numpy as np
import pandas as pd

np.random.seed(42)

n = 10000

age = np.random.randint(18, 75, n)
salary = np.random.randint(20000, 250000, n)

monthly_income = salary / 12

monthly_expenses = np.random.uniform(
    monthly_income * 0.35,
    monthly_income * 0.90
)

savings = np.random.randint(500, 500000, n)

debt = np.random.randint(0, 150000, n)

dependents = np.random.randint(0, 5, n)

investment_horizon = np.random.randint(1, 40, n)

risk_score = np.random.randint(1, 11, n)

savings_rate = (monthly_income - monthly_expenses) / monthly_income

emergency_fund_months = savings / monthly_expenses

debt_to_income = debt / salary


def assign_profile(
    age,
    savings_rate,
    emergency_fund_months,
    debt_to_income,
    investment_horizon,
    risk_score,
    dependents
):
    score = 0

    # Risk tolerance
    if risk_score <= 3:
        score -= 3
    elif risk_score <= 7:
        score += 0
    else:
        score += 3

    # Age
    if age < 30:
        score += 2
    elif age < 45:
        score += 1
    elif age < 60:
        score -= 1
    else:
        score -= 2

    # Investment horizon
    if investment_horizon >= 20:
        score += 2
    elif investment_horizon >= 10:
        score += 1
    elif investment_horizon < 5:
        score -= 2

    # Emergency fund
    if emergency_fund_months < 3:
        score -= 3
    elif emergency_fund_months >= 6:
        score += 1

    # Debt burden
    if debt_to_income > 0.6:
        score -= 3
    elif debt_to_income > 0.3:
        score -= 1

    # Savings rate
    if savings_rate > 0.30:
        score += 2
    elif savings_rate < 0.10:
        score -= 2

    # Dependents
    if dependents >= 3:
        score -= 1

    if score <= -2:
        return "Conservative"
    elif score <= 3:
        return "Balanced"
    else:
        return "Aggressive"


df = pd.DataFrame({
    "age": age,
    "salary": salary.round(2),
    "monthly_expenses": monthly_expenses.round(2),
    "savings": savings,
    "debt": debt,
    "dependents": dependents,
    "investment_horizon": investment_horizon,
    "risk_score": risk_score,
    "savings_rate": savings_rate.round(3),
    "emergency_fund_months": emergency_fund_months.round(2),
    "debt_to_income": debt_to_income.round(3)
})

df["profile"] = df.apply(
    lambda row: assign_profile(
        row["age"],
        row["savings_rate"],
        row["emergency_fund_months"],
        row["debt_to_income"],
        row["investment_horizon"],
        row["risk_score"],
        row["dependents"]
    ),
    axis=1
)

df.to_csv("data/synthetic_financial_data.csv", index=False)

print("More realistic dataset created successfully.")
print(df.shape)
print(df.head())
print(df["profile"].value_counts())