import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
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
# 1. LOAD DATA
# ============================================================

DATA_PATH = "results/feature_engineering/online_retail_final_ml_dataset.csv"

data = pd.read_csv(DATA_PATH)

print("=" * 70)
print("ONLINE RETAIL - EXPLAINABLE HYBRID ENSEMBLE MODEL")
print("=" * 70)

print("\nDataset shape:", data.shape)

print("\nColumns:")
print(list(data.columns))


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = data.drop(columns=["RepeatCustomer"])
y = data["RepeatCustomer"]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(
    y.value_counts(normalize=True) * 100
)


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

# Development data is further divided into
# training and validation data.

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    random_state=42,
    stratify=y_dev
)

print("\nDevelopment data:", X_dev.shape)
print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Test data:", X_test.shape)


# ============================================================
# 4. IDENTIFY COLUMN TYPES
# ============================================================

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

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# 5. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 6. CREATE BASE MODELS
# ============================================================

lr = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced"
        )
    )
])


rf = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            class_weight="balanced"
        )
    )
])


xgb = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42
        )
    )
])


# ============================================================
# 7. TRAIN BASE MODELS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING BASE MODELS")
print("=" * 70)

print("\nTraining Logistic Regression...")
lr.fit(X_train, y_train)

print("Training Random Forest...")
rf.fit(X_train, y_train)

print("Training XGBoost...")
xgb.fit(X_train, y_train)

print("\nAll base models trained successfully.")


# ============================================================
# 8. VALIDATION PREDICTIONS
# ============================================================

lr_val_prob = lr.predict_proba(X_val)[:, 1]
rf_val_prob = rf.predict_proba(X_val)[:, 1]
xgb_val_prob = xgb.predict_proba(X_val)[:, 1]


# ============================================================
# 9. VALIDATION ROC-AUC
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

print(
    f"Logistic Regression ROC-AUC : "
    f"{lr_val_auc:.4f}"
)

print(
    f"Random Forest ROC-AUC       : "
    f"{rf_val_auc:.4f}"
)

print(
    f"XGBoost ROC-AUC             : "
    f"{xgb_val_auc:.4f}"
)


# ============================================================
# 10. CALCULATE EHEM WEIGHTS
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
print("VALIDATION-BASED EHEM WEIGHTS")
print("=" * 70)

print(
    f"Logistic Regression weight : "
    f"{lr_weight:.4f}"
)

print(
    f"Random Forest weight       : "
    f"{rf_weight:.4f}"
)

print(
    f"XGBoost weight             : "
    f"{xgb_weight:.4f}"
)

print(
    f"\nWeight sum: "
    f"{weights.sum():.4f}"
)


# ============================================================
# 11. HYBRID VALIDATION PREDICTION
# ============================================================

hybrid_val_prob = (
    lr_weight * lr_val_prob
    + rf_weight * rf_val_prob
    + xgb_weight * xgb_val_prob
)


# ============================================================
# 12. THRESHOLD TUNING
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD SELECTION")
print("=" * 70)

thresholds = np.arange(
    0.10,
    0.91,
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


print(
    f"Best threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Validation F1  : "
    f"{best_f1:.4f}"
)


# ============================================================
# 13. TEST PREDICTIONS
# ============================================================

lr_test_prob = lr.predict_proba(X_test)[:, 1]

rf_test_prob = rf.predict_proba(X_test)[:, 1]

xgb_test_prob = xgb.predict_proba(X_test)[:, 1]


# ============================================================
# 14. EHEM TEST PROBABILITY
# ============================================================

hybrid_test_prob = (
    lr_weight * lr_test_prob
    + rf_weight * rf_test_prob
    + xgb_weight * xgb_test_prob
)


# ============================================================
# 15. FINAL PREDICTION
# ============================================================

hybrid_test_pred = (
    hybrid_test_prob >= best_threshold
).astype(int)


# ============================================================
# 16. PERFORMANCE METRICS
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

cm = confusion_matrix(
    y_test,
    hybrid_test_pred
)


# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ONLINE RETAIL EHEM TEST RESULTS")
print("=" * 70)

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
# 18. SAVE EHEM RESULTS
# ============================================================

results = pd.DataFrame({

    "Dataset": [
        "Online Retail"
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

    "XGBoost_Weight": [
        xgb_weight
    ]

})


results.to_csv(
    "results/models/online_retail_hybrid_results.csv",
    index=False
)


# ============================================================
# 19. SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual 0",
        "Actual 1"
    ],
    columns=[
        "Predicted 0",
        "Predicted 1"
    ]
)

cm_df.to_csv(
    "results/models/"
    "online_retail_hybrid_confusion_matrix.csv"
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("ONLINE RETAIL EHEM IMPLEMENTATION COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(
    "results/models/"
    "online_retail_hybrid_results.csv"
)

print(
    "results/models/"
    "online_retail_hybrid_confusion_matrix.csv"
)