import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)

from xgboost import XGBClassifier


# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv("data/synthetic_financial_data.csv")

features = [
    "age",
    "salary",
    "monthly_expenses",
    "savings",
    "debt",
    "dependents",
    "investment_horizon",
    "risk_score",
    "savings_rate",
    "emergency_fund_months",
    "debt_to_income"
]

X = df[features]
y = df["profile"]


# =========================
# 2. Encode Target Labels
# =========================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Label mapping:")
for label, encoded in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
    print(f"{label}: {encoded}")


# =========================
# 3. Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# =========================
# 4. Define Base Models
# =========================

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

gb_model = GradientBoostingClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)


# =========================
# 5. Weighted Voting Ensemble
# =========================

ensemble_model = VotingClassifier(
    estimators=[
        ("random_forest", rf_model),
        ("gradient_boosting", gb_model),
        ("xgboost", xgb_model)
    ],
    voting="soft",
    weights=[2, 2, 3]
)


# =========================
# 6. Train Individual Models
# =========================

models = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model,
    "Weighted Ensemble": ensemble_model
}

results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )
    )


# =========================
# 7. Save Best Model
# =========================

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nModel Comparison:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")

print(f"\nBest Model: {best_model_name}")

joblib.dump(best_model, "models/investor_profile_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
joblib.dump(features, "models/features.pkl")

print("\nBest model saved successfully.")
print("Saved files:")
print("- models/investor_profile_model.pkl")
print("- models/label_encoder.pkl")
print("- models/features.pkl")