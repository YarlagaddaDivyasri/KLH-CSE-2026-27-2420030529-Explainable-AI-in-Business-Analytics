import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\DELL\OneDrive\Desktop\Business Analytics"

DATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "feature_engineering",
    "online_retail_final_ml_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "online_retail_models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "models"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 80)
print("ONLINE RETAIL - FINAL TEST EVALUATION")
print("=" * 80)


df = pd.read_csv(
    DATA_PATH
)

TARGET = "RepeatCustomer"

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET].astype(int)


# ============================================================
# 3. REMOVE IDENTIFIERS
# ============================================================

identifier_columns = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "CustomerID"
]

columns_to_remove = [
    col
    for col in identifier_columns
    if col in X.columns
]

if columns_to_remove:

    X = X.drop(
        columns=columns_to_remove
    )


# ============================================================
# 4. SAME SPLIT
# ============================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    stratify=y_dev,
    random_state=42
)


print("\nData split:")

print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ============================================================
# 5. LOAD MODELS
# ============================================================

rf_path = os.path.join(
    MODEL_DIR,
    "random_forest.joblib"
)

hgb_path = os.path.join(
    MODEL_DIR,
    "histgradientboosting.joblib"
)


print("\nLoading models...")


rf_model = joblib.load(
    rf_path
)

hgb_model = joblib.load(
    hgb_path
)


# ============================================================
# 6. GENERATE TEST PROBABILITIES
# ============================================================

print(
    "\nGenerating predictions on untouched test set..."
)


rf_test_probability = (
    rf_model.predict_proba(
        X_test
    )[:, 1]
)


hgb_test_probability = (
    hgb_model.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 7. LOCKED CONFIGURATION
# ============================================================

RF_WEIGHT = 0.30

HGB_WEIGHT = 0.70

THRESHOLD = 0.64


print("\n" + "=" * 80)
print("LOCKED HYBRID CONFIGURATION")
print("=" * 80)

print(
    "RF weight:",
    RF_WEIGHT
)

print(
    "HGB weight:",
    HGB_WEIGHT
)

print(
    "Threshold:",
    THRESHOLD
)


# ============================================================
# 8. HYBRID PROBABILITY
# ============================================================

hybrid_probability = (
    RF_WEIGHT * rf_test_probability
    +
    HGB_WEIGHT * hgb_test_probability
)


# ============================================================
# 9. FINAL PREDICTION
# ============================================================

hybrid_prediction = (
    hybrid_probability >= THRESHOLD
).astype(int)


# ============================================================
# 10. HYBRID METRICS
# ============================================================

hybrid_accuracy = accuracy_score(
    y_test,
    hybrid_prediction
)

hybrid_precision = precision_score(
    y_test,
    hybrid_prediction,
    zero_division=0
)

hybrid_recall = recall_score(
    y_test,
    hybrid_prediction,
    zero_division=0
)

hybrid_f1 = f1_score(
    y_test,
    hybrid_prediction,
    zero_division=0
)

hybrid_auc = roc_auc_score(
    y_test,
    hybrid_probability
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    hybrid_prediction
)


# ============================================================
# 12. DISPLAY HYBRID RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL RETAIL HYBRID TEST RESULTS")
print("=" * 80)


print(
    "\nAccuracy :",
    f"{hybrid_accuracy:.6f}"
)

print(
    "Precision:",
    f"{hybrid_precision:.6f}"
)

print(
    "Recall   :",
    f"{hybrid_recall:.6f}"
)

print(
    "F1       :",
    f"{hybrid_f1:.6f}"
)

print(
    "ROC-AUC  :",
    f"{hybrid_auc:.6f}"
)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 13. SAVE FINAL HYBRID RESULTS
# ============================================================

final_result = pd.DataFrame([
    {
        "Model": "Optimized RF + HGB",
        "RF_Weight": RF_WEIGHT,
        "HGB_Weight": HGB_WEIGHT,
        "Threshold": THRESHOLD,
        "Accuracy": hybrid_accuracy,
        "Precision": hybrid_precision,
        "Recall": hybrid_recall,
        "F1": hybrid_f1,
        "ROC-AUC": hybrid_auc
    }
])


final_result_path = os.path.join(
    RESULTS_DIR,
    "epe_final_test_retail.csv"
)


final_result.to_csv(
    final_result_path,
    index=False
)


# ============================================================
# 14. SAVE CONFUSION MATRIX
# ============================================================

confusion_df = pd.DataFrame(
    cm,
    index=[
        "Actual_0",
        "Actual_1"
    ],
    columns=[
        "Predicted_0",
        "Predicted_1"
    ]
)


confusion_path = os.path.join(
    RESULTS_DIR,
    "epe_h3_confusion_matrix_retail.csv"
)


confusion_df.to_csv(
    confusion_path
)


# ============================================================
# 15. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual":
        y_test.reset_index(
            drop=True
        ),

    "RF_Probability":
        rf_test_probability,

    "HGB_Probability":
        hgb_test_probability,

    "Hybrid_Probability":
        hybrid_probability,

    "Prediction":
        hybrid_prediction
})


prediction_path = os.path.join(
    RESULTS_DIR,
    "epe_h3_test_predictions_retail.csv"
)


prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# 16. SAVE MODEL COMPARISON
# ============================================================

baseline_path = os.path.join(
    RESULTS_DIR,
    "online_retail_baseline_models.csv"
)


if os.path.exists(baseline_path):

    baseline_df = pd.read_csv(
        baseline_path
    )

    comparison_rows = []

    for _, row in baseline_df.iterrows():

        comparison_rows.append({

            "Model":
                row["Model"],

            "Accuracy":
                row["Accuracy"],

            "Precision":
                row["Precision"],

            "Recall":
                row["Recall"],

            "F1":
                row["F1"],

            "ROC-AUC":
                row["ROC-AUC"]
        })


    comparison_rows.append({

        "Model":
            "Optimized RF + HGB",

        "Accuracy":
            hybrid_accuracy,

        "Precision":
            hybrid_precision,

        "Recall":
            hybrid_recall,

        "F1":
            hybrid_f1,

        "ROC-AUC":
            hybrid_auc
    })


    comparison_df = pd.DataFrame(
        comparison_rows
    )


    comparison_path = os.path.join(
        RESULTS_DIR,
        "epe_final_test_comparison_retail.csv"
    )


    comparison_df.to_csv(
        comparison_path,
        index=False
    )


else:

    comparison_path = None


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)


print(
    "\n1.",
    final_result_path
)

print(
    "2.",
    confusion_path
)

print(
    "3.",
    prediction_path
)

if comparison_path:

    print(
        "4.",
        comparison_path
    )


print("\n" + "=" * 80)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 80)

print(
    "\nThe test set was evaluated using the configuration "
    "selected only from validation data."
)

print(
    "\nRF weight = 0.30"
)

print(
    "HGB weight = 0.70"
)

print(
    "Threshold = 0.64"
)