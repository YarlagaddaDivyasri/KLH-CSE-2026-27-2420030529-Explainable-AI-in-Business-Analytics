import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\DELL\OneDrive\Desktop\Business Analytics"

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "models"
)

MODEL_DIR = os.path.join(
    RESULTS_DIR,
    "online_retail_models"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "feature_engineering",
    "online_retail_final_ml_dataset.csv"
)

HYBRID_PATH = os.path.join(
    RESULTS_DIR,
    "online_retail_hybrid_comparison.csv"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 80)
print("ONLINE RETAIL - RF + HGB OPTIMIZATION")
print("=" * 80)


df = pd.read_csv(DATA_PATH)

TARGET = "RepeatCustomer"

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET].astype(int)


# ============================================================
# 3. IDENTIFIER REMOVAL
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
# 4. SAME SPLIT AS PREVIOUS IMPLEMENTATION
# ============================================================

from sklearn.model_selection import train_test_split


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
# 5. LOAD TRAINED RF AND HGB MODELS
# ============================================================

rf_path = os.path.join(
    MODEL_DIR,
    "random_forest.joblib"
)

hgb_path = os.path.join(
    MODEL_DIR,
    "histgradientboosting.joblib"
)


if not os.path.exists(rf_path):

    raise FileNotFoundError(
        f"Random Forest model not found:\n{rf_path}"
    )


if not os.path.exists(hgb_path):

    raise FileNotFoundError(
        f"HistGradientBoosting model not found:\n{hgb_path}"
    )


print("\nLoading trained models...")

rf_model = joblib.load(
    rf_path
)

hgb_model = joblib.load(
    hgb_path
)


# ============================================================
# 6. GET VALIDATION AND TEST PROBABILITIES
# ============================================================

print("\nGenerating probabilities...")


rf_val_probability = (
    rf_model.predict_proba(X_val)[:, 1]
)

hgb_val_probability = (
    hgb_model.predict_proba(X_val)[:, 1]
)


rf_test_probability = (
    rf_model.predict_proba(X_test)[:, 1]
)

hgb_test_probability = (
    hgb_model.predict_proba(X_test)[:, 1]
)


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {

        "Accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "Precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "ROC-AUC":
            roc_auc_score(
                y_true,
                probabilities
            )
    }


# ============================================================
# 8. WEIGHT + THRESHOLD SEARCH
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION-ONLY OPTIMIZATION")
print("=" * 80)


optimization_results = []


# RF weight
# HGB weight = 1 - RF weight

rf_weights = np.arange(
    0.10,
    0.91,
    0.05
)


thresholds = np.arange(
    0.20,
    0.71,
    0.01
)


experiment_number = 0


for rf_weight in rf_weights:

    hgb_weight = (
        1.0 - rf_weight
    )


    # Weighted validation probability

    validation_probability = (
        rf_weight * rf_val_probability
        +
        hgb_weight * hgb_val_probability
    )


    roc_auc = roc_auc_score(
        y_val,
        validation_probability
    )


    for threshold in thresholds:

        experiment_number += 1


        predictions = (
            validation_probability
            >= threshold
        ).astype(int)


        accuracy = accuracy_score(
            y_val,
            predictions
        )


        precision = precision_score(
            y_val,
            predictions,
            zero_division=0
        )


        recall = recall_score(
            y_val,
            predictions,
            zero_division=0
        )


        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0
        )


        optimization_results.append({

            "RF_Weight":
                rf_weight,

            "HGB_Weight":
                hgb_weight,

            "Threshold":
                threshold,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                f1,

            "ROC-AUC":
                roc_auc
        })


print(
    "\nTotal experiments:",
    experiment_number
)


# ============================================================
# 9. SAVE ALL SEARCH RESULTS
# ============================================================

optimization_df = pd.DataFrame(
    optimization_results
)


search_path = os.path.join(
    RESULTS_DIR,
    "online_retail_rf_hgb_optimization_search.csv"
)


optimization_df.to_csv(
    search_path,
    index=False
)


# ============================================================
# 10. BEST CONFIGURATION BY F1
# ============================================================

best_f1_row = (
    optimization_df
    .sort_values(
        by=[
            "F1",
            "ROC-AUC"
        ],
        ascending=False
    )
    .iloc[0]
)


# ============================================================
# 11. BEST CONFIGURATION BY ROC-AUC
# ============================================================

best_auc_row = (
    optimization_df
    .sort_values(
        by=[
            "ROC-AUC",
            "F1"
        ],
        ascending=False
    )
    .iloc[0]
)


# ============================================================
# 12. TOP 10 CONFIGURATIONS
# ============================================================

top10_df = (
    optimization_df
    .sort_values(
        by=[
            "F1",
            "ROC-AUC"
        ],
        ascending=False
    )
    .head(10)
)


top10_path = os.path.join(
    RESULTS_DIR,
    "online_retail_rf_hgb_optimization_top10.csv"
)


top10_df.to_csv(
    top10_path,
    index=False
)


# ============================================================
# 13. DISPLAY BEST F1 CONFIGURATION
# ============================================================

print("\n" + "=" * 80)
print("BEST CONFIGURATION BY VALIDATION F1")
print("=" * 80)


print(
    "\nRF Weight:",
    round(
        best_f1_row["RF_Weight"],
        4
    )
)


print(
    "HGB Weight:",
    round(
        best_f1_row["HGB_Weight"],
        4
    )
)


print(
    "Threshold:",
    round(
        best_f1_row["Threshold"],
        4
    )
)


print(
    "Accuracy:",
    round(
        best_f1_row["Accuracy"],
        6
    )
)


print(
    "Precision:",
    round(
        best_f1_row["Precision"],
        6
    )
)


print(
    "Recall:",
    round(
        best_f1_row["Recall"],
        6
    )
)


print(
    "F1:",
    round(
        best_f1_row["F1"],
        6
    )
)


print(
    "ROC-AUC:",
    round(
        best_f1_row["ROC-AUC"],
        6
    )
)


# ============================================================
# 14. DISPLAY BEST AUC CONFIGURATION
# ============================================================

print("\n" + "=" * 80)
print("BEST CONFIGURATION BY VALIDATION ROC-AUC")
print("=" * 80)


print(
    "\nRF Weight:",
    round(
        best_auc_row["RF_Weight"],
        4
    )
)


print(
    "HGB Weight:",
    round(
        best_auc_row["HGB_Weight"],
        4
    )
)


print(
    "Threshold:",
    round(
        best_auc_row["Threshold"],
        4
    )
)


print(
    "Accuracy:",
    round(
        best_auc_row["Accuracy"],
        6
    )
)


print(
    "Precision:",
    round(
        best_auc_row["Precision"],
        6
    )
)


print(
    "Recall:",
    round(
        best_auc_row["Recall"],
        6
    )
)


print(
    "F1:",
    round(
        best_auc_row["F1"],
        6
    )
)


print(
    "ROC-AUC:",
    round(
        best_auc_row["ROC-AUC"],
        6
    )
)


# ============================================================
# 15. SAVE BEST CONFIGURATION
# ============================================================

best_config = {

    "Hybrid":
        "RF + HGB",

    "RF_Weight":
        float(
            best_f1_row["RF_Weight"]
        ),

    "HGB_Weight":
        float(
            best_f1_row["HGB_Weight"]
        ),

    "Threshold":
        float(
            best_f1_row["Threshold"]
        ),

    "Validation_Accuracy":
        float(
            best_f1_row["Accuracy"]
        ),

    "Validation_Precision":
        float(
            best_f1_row["Precision"]
        ),

    "Validation_Recall":
        float(
            best_f1_row["Recall"]
        ),

    "Validation_F1":
        float(
            best_f1_row["F1"]
        ),

    "Validation_ROC_AUC":
        float(
            best_f1_row["ROC-AUC"]
        )
}


best_config_df = pd.DataFrame(
    [best_config]
)


best_config_path = os.path.join(
    RESULTS_DIR,
    "online_retail_rf_hgb_best_config.csv"
)


best_config_df.to_csv(
    best_config_path,
    index=False
)


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETED")
print("=" * 80)


print(
    "\nSaved:"
)

print(
    "1.",
    search_path
)

print(
    "2.",
    top10_path
)

print(
    "3.",
    best_config_path
)


print(
    "\nIMPORTANT:"
)

print(
    "The test set has NOT been used for selecting the "
    "weight or threshold."
)

print(
    "\nNext step:"
)

print(
    "Run the selected RF + HGB configuration once on "
    "the untouched test set."
)