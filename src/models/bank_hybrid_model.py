import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "results/feature_selection/bank_selected_features.csv"

data = pd.read_csv(DATA_PATH)

print("=" * 70)
print("BANK MARKETING - EXPLAINABLE HYBRID ENSEMBLE MODEL")
print("=" * 70)

print("\nDataset shape:", data.shape)

print("\nColumns:")
print(list(data.columns))


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["y"])
y = data["y"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 3. DEVELOPMENT / TEST SPLIT
# ============================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDevelopment data:", X_dev.shape)
print("Test data:", X_test.shape)


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


# ============================================================
# 5. DEFINE BASE MODELS
# ============================================================

# Logistic Regression
lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        solver="liblinear"
    ))
])


# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)


# ============================================================
# 6. TRAIN BASE MODELS
# ============================================================

print("\nTraining Logistic Regression...")
lr.fit(X_train, y_train)

print("Training Random Forest...")
rf.fit(X_train, y_train)

print("Training XGBoost...")
xgb.fit(X_train, y_train)

print("\nAll base models trained successfully.")


# ============================================================
# 7. VALIDATION PREDICTIONS
# ============================================================

lr_val_prob = lr.predict_proba(X_val)[:, 1]

rf_val_prob = rf.predict_proba(X_val)[:, 1]

xgb_val_prob = xgb.predict_proba(X_val)[:, 1]


# ============================================================
# 8. CALCULATE VALIDATION ROC-AUC
# ============================================================

lr_val_auc = roc_auc_score(
    y_val,
    lr_val_prob
)

rf_val_auc = roc_auc_score(
    y_val,
    rf_val_prob
)

xgb_val_auc = roc_auc_score(
    y_val,
    xgb_val_prob
)


print("\n" + "=" * 70)
print("VALIDATION PERFORMANCE")
print("=" * 70)

print(f"Logistic Regression ROC-AUC : {lr_val_auc:.4f}")
print(f"Random Forest ROC-AUC       : {rf_val_auc:.4f}")
print(f"XGBoost ROC-AUC             : {xgb_val_auc:.4f}")


# ============================================================
# 9. CALCULATE VALIDATION-BASED WEIGHTS
# ============================================================

scores = np.array([
    lr_val_auc,
    rf_val_auc,
    xgb_val_auc
])

weights = scores / scores.sum()

lr_weight = weights[0]
rf_weight = weights[1]
xgb_weight = weights[2]


print("\n" + "=" * 70)
print("VALIDATION-BASED MODEL WEIGHTS")
print("=" * 70)

print(f"Logistic Regression weight : {lr_weight:.4f}")
print(f"Random Forest weight       : {rf_weight:.4f}")
print(f"XGBoost weight             : {xgb_weight:.4f}")

print("\nWeight sum:", weights.sum())


# ============================================================
# 10. CREATE HYBRID VALIDATION PROBABILITY
# ============================================================

hybrid_val_prob = (
    lr_weight * lr_val_prob
    + rf_weight * rf_val_prob
    + xgb_weight * xgb_val_prob
)


# ============================================================
# 11. THRESHOLD SELECTION
# ============================================================

thresholds = np.arange(
    0.30,
    0.71,
    0.01
)

best_threshold = 0.50
best_f1 = -1

for threshold in thresholds:

    val_pred = (
        hybrid_val_prob >= threshold
    ).astype(int)

    current_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold


print("\n" + "=" * 70)
print("THRESHOLD SELECTION")
print("=" * 70)

print(f"Best threshold : {best_threshold:.2f}")
print(f"Validation F1  : {best_f1:.4f}")


# ============================================================
# 12. TEST PREDICTIONS
# ============================================================

lr_test_prob = lr.predict_proba(X_test)[:, 1]

rf_test_prob = rf.predict_proba(X_test)[:, 1]

xgb_test_prob = xgb.predict_proba(X_test)[:, 1]


# ============================================================
# 13. HYBRID TEST PROBABILITY
# ============================================================

hybrid_test_prob = (
    lr_weight * lr_test_prob
    + rf_weight * rf_test_prob
    + xgb_weight * xgb_test_prob
)


# ============================================================
# 14. FINAL HYBRID PREDICTION
# ============================================================

hybrid_test_pred = (
    hybrid_test_prob >= best_threshold
).astype(int)


# ============================================================
# 15. CALCULATE TEST METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    hybrid_test_pred
)

precision = precision_score(
    y_test,
    hybrid_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    hybrid_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    hybrid_test_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    hybrid_test_prob
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    hybrid_test_pred
)


# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EHEM TEST RESULTS")
print("=" * 70)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 18. SAVE MODEL RESULTS
# ============================================================

results = pd.DataFrame({
    "Dataset": ["Bank Marketing"],
    "Model": ["EHEM"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1-Score": [f1],
    "ROC-AUC": [roc_auc],
    "Threshold": [best_threshold],
    "LR_Weight": [lr_weight],
    "RF_Weight": [rf_weight],
    "XGBoost_Weight": [xgb_weight]
})

results.to_csv(
    "results/models/bank_hybrid_results.csv",
    index=False
)


# ============================================================
# 19. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)

cm_df.to_csv(
    "results/models/bank_hybrid_confusion_matrix.csv"
)


print("\nResults saved successfully.")

print("\nFiles created:")
print("results/models/bank_hybrid_results.csv")
print("results/models/bank_hybrid_confusion_matrix.csv")

print("\n" + "=" * 70)
print("BANK EHEM IMPLEMENTATION COMPLETED")
print("=" * 70)