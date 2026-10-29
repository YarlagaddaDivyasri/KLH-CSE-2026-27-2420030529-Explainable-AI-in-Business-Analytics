import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "feature_selection",
    "telco_selected_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "models"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

TARGET = "Churn"

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]

print("\nFeatures:")
print(list(X.columns))

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    random_state=42,
    stratify=y_dev
)

print("\nTraining data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Test data:", X_test.shape)


# ============================================================
# 5. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 6. TRAIN BASE MODELS
# ============================================================

print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear",
    random_state=42
)

lr.fit(
    X_train_scaled,
    y_train
)


print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train_scaled,
    y_train
)


print("Training XGBoost...")

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 7. VALIDATION PREDICTIONS
# ============================================================

lr_val_prob = lr.predict_proba(
    X_val_scaled
)[:, 1]

rf_val_prob = rf.predict_proba(
    X_val_scaled
)[:, 1]

xgb_val_prob = xgb.predict_proba(
    X_val_scaled
)[:, 1]


# ============================================================
# 8. VALIDATION ROC-AUC
# ============================================================

lr_auc = roc_auc_score(
    y_val,
    lr_val_prob
)

rf_auc = roc_auc_score(
    y_val,
    rf_val_prob
)

xgb_auc = roc_auc_score(
    y_val,
    xgb_val_prob
)


print("\nValidation ROC-AUC:")

print(
    f"Logistic Regression : {lr_auc:.4f}"
)

print(
    f"Random Forest       : {rf_auc:.4f}"
)

print(
    f"XGBoost             : {xgb_auc:.4f}"
)


# ============================================================
# 9. CALCULATE EHEM WEIGHTS
# ============================================================

auc_values = np.array([
    lr_auc,
    rf_auc,
    xgb_auc
])

weights = (
    auc_values /
    auc_values.sum()
)

lr_weight = weights[0]
rf_weight = weights[1]
xgb_weight = weights[2]


print("\nEHEM Weights:")

print(
    f"Logistic Regression : "
    f"{lr_weight:.4f}"
)

print(
    f"Random Forest       : "
    f"{rf_weight:.4f}"
)

print(
    f"XGBoost             : "
    f"{xgb_weight:.4f}"
)


# ============================================================
# 10. HYBRID PROBABILITY
# ============================================================

def hybrid_probability(
    lr_prob,
    rf_prob,
    xgb_prob
):

    return (
        lr_weight * lr_prob
        +
        rf_weight * rf_prob
        +
        xgb_weight * xgb_prob
    )


validation_hybrid_prob = hybrid_probability(
    lr_val_prob,
    rf_val_prob,
    xgb_val_prob
)


# ============================================================
# 11. THRESHOLD TUNING
# ============================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.01
)

best_threshold = 0.50
best_f1 = 0


for threshold in thresholds:

    val_prediction = (
        validation_hybrid_prob
        >= threshold
    ).astype(int)

    current_f1 = f1_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    if current_f1 > best_f1:

        best_f1 = current_f1
        best_threshold = threshold


print("\nThreshold tuning:")

print(
    f"Best threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Validation F1  : "
    f"{best_f1:.4f}"
)


# ============================================================
# 12. TEST PREDICTIONS
# ============================================================

lr_test_prob = lr.predict_proba(
    X_test_scaled
)[:, 1]

rf_test_prob = rf.predict_proba(
    X_test_scaled
)[:, 1]

xgb_test_prob = xgb.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# 13. EHEM TEST PROBABILITY
# ============================================================

ehem_test_prob = hybrid_probability(
    lr_test_prob,
    rf_test_prob,
    xgb_test_prob
)


# ============================================================
# 14. APPLY VALIDATION THRESHOLD
# ============================================================

ehem_test_pred = (
    ehem_test_prob
    >= best_threshold
).astype(int)


# ============================================================
# 15. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    ehem_test_pred
)

precision = precision_score(
    y_test,
    ehem_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    ehem_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    ehem_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    ehem_test_prob
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    ehem_test_pred
)


print("\n======================================")
print("TELCO EHEM TEST RESULTS")
print("======================================")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# 17. SAVE MODEL RESULTS
# ============================================================

results_df = pd.DataFrame({

    "Dataset": [
        "Telco Customer Churn"
    ],

    "Model": [
        "EHEM"
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1-Score": [
        f1
    ],

    "ROC-AUC": [
        roc_auc
    ],

    "Threshold": [
        best_threshold
    ],

    "LR_Weight": [
        lr_weight
    ],

    "RF_Weight": [
        rf_weight
    ],

    "XGB_Weight": [
        xgb_weight
    ]
})


results_path = os.path.join(
    OUTPUT_DIR,
    "telco_hybrid_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)


print(
    "\nResults saved to:"
)

print(results_path)


# ============================================================
# 18. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
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

cm_path = os.path.join(
    OUTPUT_DIR,
    "telco_hybrid_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


print(
    "Confusion matrix saved to:"
)

print(cm_path)


# ============================================================
# 19. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "Actual": y_test.values,

    "EHEM_Probability":
        ehem_test_prob,

    "EHEM_Prediction":
        ehem_test_pred
})


prediction_path = os.path.join(
    OUTPUT_DIR,
    "telco_hybrid_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


print(
    "Predictions saved to:"
)

print(prediction_path)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n======================================")
print("TELCO EHEM IMPLEMENTATION COMPLETED")
print("======================================")