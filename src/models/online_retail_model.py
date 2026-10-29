import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# 1. Load final Online Retail ML dataset
data = pd.read_csv(
    "results/feature_engineering/online_retail_final_ml_dataset.csv"
)

print("Dataset shape:", data.shape)


# 2. Separate features and target
X = data.drop("RepeatCustomer", axis=1)
y = data["RepeatCustomer"]


# 3. Identify categorical and numerical columns
categorical_features = ["Country"]

numerical_features = [
    "Quantity",
    "UnitPrice",
    "Year",
    "Month",
    "Day",
    "Hour",
    "IsReturn"
]


# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# 6. Define models
models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# 7. Create results folder
os.makedirs("results/models", exist_ok=True)


results = []


# 8. Train and evaluate each model
for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # Save confusion matrix
    safe_name = name.lower().replace(" ", "_")

    cm_file = (
        f"results/models/"
        f"online_retail_{safe_name}_confusion_matrix.csv"
    )

    pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    ).to_csv(cm_file)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("Confusion Matrix:")
    print(cm)


# 9. Save model comparison
results_df = pd.DataFrame(results)

results_file = (
    "results/models/"
    "online_retail_model_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# 10. Display final comparison
print("\n" + "=" * 70)
print("ONLINE RETAIL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)

print("\nResults saved to:")
print(results_file)