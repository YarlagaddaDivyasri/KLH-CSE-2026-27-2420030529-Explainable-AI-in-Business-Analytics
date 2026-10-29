import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

data_path = r"results\feature_selection\bank_selected_features.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("BANK MARKETING ML MODEL IMPLEMENTATION")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("y", axis=1)
y = df["y"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 4. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
    max_iter=5000,
    solver="liblinear",
    random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
}


# ============================================================
# 5. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

confusion_matrices = {}


for model_name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Probability for ROC-AUC
    y_probability = model.predict_proba(X_test)[:, 1]

    # Evaluation metrics
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
        y_probability
    )

    cm = confusion_matrix(y_test, y_pred)

    confusion_matrices[model_name] = cm

    # Store results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    # Print results
    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nConfusion Matrix:")
    print(cm)


# ============================================================
# 6. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(results_df)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

output_dir = r"results\models"

os.makedirs(output_dir, exist_ok=True)

results_path = os.path.join(
    output_dir,
    "bank_model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nResults saved to:")
print(results_path)


# ============================================================
# 8. SAVE CONFUSION MATRICES
# ============================================================

for model_name, cm in confusion_matrices.items():

    safe_name = model_name.lower().replace(" ", "_")

    cm_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    cm_path = os.path.join(
        output_dir,
        f"bank_{safe_name}_confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

print("\nConfusion matrices saved.")


# ============================================================
# 9. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("BANK ML MODEL IMPLEMENTATION COMPLETED")
print("=" * 60)
# ============================================================
# 10. PERFORMANCE COMPARISON GRAPHS
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

for metric in metrics:

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Model"],
        results_df[metric]
    )

    plt.title(f"Bank Marketing - {metric} Comparison")
    plt.xlabel("Machine Learning Model")
    plt.ylabel(metric)

    plt.ylim(0, 1)

    plt.xticks(rotation=15)

    plt.tight_layout()

    graph_path = os.path.join(
        output_dir,
        f"bank_{metric.lower().replace('-', '_')}_comparison.png"
    )

    plt.savefig(graph_path, dpi=300)

    plt.show()

    print(f"{metric} graph saved to:")
    print(graph_path)