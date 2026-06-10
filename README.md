# AI Financial Advisor

## Overview

This project is an AI-powered Financial Advisor web application built using Machine Learning, Ensemble Learning, Streamlit, and Google's Gemini AI.

The goal of the project is to demonstrate how machine learning can be used to classify investors into different risk profiles and generate portfolio allocations based on their financial situation.

Users provide information such as income, savings, expenses, debt levels, investment horizon, and risk tolerance. The application then predicts an investor profile and generates an example diversified portfolio allocation across multiple asset classes.

This project was developed as a portfolio project to showcase practical applications of machine learning in financial technology (FinTech).

---

## Features

### Investor Profiling

The application analyzes user financial information and classifies investors into one of three categories:

* Conservative
* Balanced
* Aggressive

### Portfolio Allocation

The system generates a diversified portfolio allocation across:

* Global Equities
* Government Bonds
* Corporate Bonds
* REITs
* Commodities
* Gold
* Cash
* Cryptocurrency

### Ensemble Machine Learning Model

The prediction engine uses a weighted ensemble of:

* Random Forest
* Gradient Boosting
* XGBoost

The ensemble achieved approximately 97% classification accuracy on the synthetic dataset used in this project.

### Risk Assessment

The application generates a risk score and visualizes it using an interactive gauge chart.

### Portfolio Projection

Users can estimate future portfolio growth based on:

* Monthly contributions
* Expected annual return
* Investment horizon

### Gemini AI Explanation

Google Gemini is used to generate a human-readable explanation of the recommendation, helping users understand why a particular portfolio was suggested.

---

## Technologies Used

### Programming Language

* Python

### Data Science & Machine Learning

* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Visualization

* Plotly

### Web Application

* Streamlit

### Large Language Model

* Google Gemini API

---

## Project Structure

```text
ai-financial-advisor-app/

├── app.py
├── generate_data.py
├── train_model.py
├── predict.py
├── risk_profiles.py
├── asset_mapping.py
│
├── data/
│   └── synthetic_financial_data.csv
│
├── models/
│   ├── investor_profile_model.pkl
│   ├── label_encoder.pkl
│   └── features.pkl
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Pipeline

### Data Generation

A synthetic financial dataset was created containing:

* Age
* Salary
* Monthly Expenses
* Savings
* Debt
* Dependents
* Investment Horizon
* Risk Score
* Savings Rate
* Emergency Fund Coverage
* Debt-to-Income Ratio

### Model Training

Three machine learning models were trained:

1. Random Forest
2. Gradient Boosting
3. XGBoost

A weighted soft-voting ensemble was then used to combine predictions from all three models.

### Model Performance

| Model             | Accuracy |
| ----------------- | -------- |
| Random Forest     | 92.75%   |
| Gradient Boosting | 96.95%   |
| XGBoost           | 97.25%   |
| Weighted Ensemble | 97.30%   |

The weighted ensemble achieved the best overall performance and was selected as the final production model.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Debodip-cloud/ai-financial-advisor-app.git

cd ai-financial-advisor-app
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Gemini Configuration

To enable AI-generated explanations, create a Gemini API key and set it as an environment variable.

Windows:

```bash
set GEMINI_API_KEY=YOUR_API_KEY
```

Linux/Mac:

```bash
export GEMINI_API_KEY=YOUR_API_KEY
```

Alternatively, when deploying to Streamlit Cloud, add the key through Streamlit Secrets.

---

## Educational Disclaimer

This project is intended for educational and demonstration purposes only.

The generated portfolio allocations and investment projections should not be interpreted as financial advice, investment advice, or recommendations to buy or sell any financial instrument.

Always consult a qualified financial professional before making investment decisions.

---

## Author

Debodip Chowdhury

MSc Financial Technology with Data Science
University of Bristol

YouTube: Infometrics Labs

GitHub: https://github.com/Debodip-cloud
