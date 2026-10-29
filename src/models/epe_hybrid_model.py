# ============================================================
# EPE HYBRID MODEL - BANK MARKETING DATASET
# ============================================================
# EPE = Explainable Performance-based Ensemble
#
# Models:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 4. XGBoost
# 5. SVM
# 6. HistGradientBoosting
#
# This file:
# - Loads selected Bank Marketing features
# - Splits data into Train / Validation / Test
# - Trains 6 individual models
# - Evaluates all 6 models on validation data
# - Saves validation results
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# 1. FIND PROJECT DIRECTORY
# ============================================================

# Current file:
# Business Analytics/backend/models/epe_hybrid_model.py
#
# Going three levels up gives:
# Business Analytics/

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

print("=" * 70)
print("EPE HYBRID MODEL - BANK MARKETING DATASET")
print("=" * 70)

print("\nProject directory:")
print(BASE_DIR)


# ============================================================
# 2. DEFINE DATASET PATH
# ============================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "feature_selection",
    "bank_selected_features.csv"
)


# ============================================================
# 3. DEFINE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "models"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 4. CHECK DATASET
# ============================================================

print("\nChecking dataset...")

if not os.path.exists(DATA_PATH):

    print("\nERROR: Dataset file was not found.")

    print("\nExpected file:")
    print(DATA_PATH)

    print("\nPlease make sure this file exists:")
    print(
        "results\\feature_selection\\bank_selected_features.csv"
    )

    input("\nPress Enter to exit...")
    exit()


print("Dataset found successfully!")


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 6. DISPLAY DATASET INFORMATION
# ============================================================

print("\nDataset columns:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 7. CHECK TARGET COLUMN
# ============================================================

TARGET_COLUMN = "y"

if TARGET_COLUMN not in df.columns:

    print("\nERROR: Target column 'y' was not found.")

    print("Available columns:")
    print(list(df.columns))

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# 8. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(TARGET_COLUMN, axis=1)

y = df[TARGET_COLUMN]


print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 9. CHECK FOR MISSING VALUES
# ============================================================

missing_values = X.isnull().sum().sum()

print("\nTotal missing values:", missing_values)

if missing_values > 0:

    print("\nWARNING: Missing values detected.")

    # Fill numerical missing values with median
    X = X.fillna(X.median(numeric_only=True))


# ============================================================
# 10. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")


# First:
# 80% temporary data
# 20% test data

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# From the 80% temporary data:
# 80% -> training
# 20% -> validation
#
# Final:
# Training     = 64%
# Validation   = 16%
# Testing      = 20%

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    random_state=42,
    stratify=y_temp
)


print("\nData split completed.")

print("Training samples   :", len(X_train))
print("Validation samples :", len(X_val))
print("Testing samples    :", len(X_test))


# ============================================================
# 11. DEFINE SIX INDIVIDUAL MODELS
# ============================================================

print("\nCreating models...")


# ------------------------------------------------------------
# Logistic Regression
# ------------------------------------------------------------

lr = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# ------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------

dt = DecisionTreeClassifier(
    random_state=42
)


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric="logloss"
)


# ------------------------------------------------------------
# Support Vector Machine
# ------------------------------------------------------------

svm = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        SVC(
            probability=True,
            random_state=42
        )
    )
])


# ------------------------------------------------------------
# HistGradientBoosting
# ------------------------------------------------------------

hgb = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)


# ============================================================
# 12. STORE MODELS IN DICTIONARY
# ============================================================

models = {

    "Logistic Regression": lr,

    "Decision Tree": dt,

    "Random Forest": rf,

    "XGBoost": xgb,

    "SVM": svm,

    "HistGradientBoosting": hgb

}


print("\nModels created:")
for name in models:
    print("-", name)


# ============================================================
# 13. TRAIN ALL SIX MODELS
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING INDIVIDUAL MODELS")
print("=" * 70)


for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train,
        y_train
    )

    print(
        name,
        "training completed."
    )


# ============================================================
# 14. EVALUATE INDIVIDUAL MODELS ON VALIDATION DATA
# ============================================================

print("\n")
print("=" * 70)
print("VALIDATION PERFORMANCE")
print("=" * 70)


individual_results = []


# Store validation probabilities
# These will be used later for hybrid models.

validation_probabilities = {}


for name, model in models.items():

    print("\nEvaluating:", name)

    # Probability of positive class
    val_probability = model.predict_proba(
        X_val
    )[:, 1]

    # Default threshold = 0.50
    val_prediction = (
        val_probability >= 0.50
    ).astype(int)


    # Store probabilities
    validation_probabilities[name] = val_probability


    # Calculate metrics

    accuracy = accuracy_score(
        y_val,
        val_prediction
    )

    precision = precision_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        val_probability
    )


    # Store results

    individual_results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "ROC-AUC": roc_auc

    })


# ============================================================
# 15. CREATE RESULTS DATAFRAME
# ============================================================

individual_results_df = pd.DataFrame(
    individual_results
)


# ============================================================
# 16. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("INDIVIDUAL MODEL VALIDATION RESULTS")
print("=" * 70)

print(
    individual_results_df.to_string(
        index=False
    )
)


# ============================================================
# 17. SAVE INDIVIDUAL MODEL RESULTS
# ============================================================

individual_results_path = os.path.join(
    RESULTS_DIR,
    "epe_individual_model_results_bank.csv"
)


individual_results_df.to_csv(
    individual_results_path,
    index=False
)


print("\nIndividual model results saved to:")

print(
    individual_results_path
)


# ============================================================
# 18. SAVE VALIDATION PROBABILITIES
# ============================================================

validation_probability_df = pd.DataFrame(
    validation_probabilities
)


validation_probability_df["Actual"] = (
    y_val.values
)


validation_probability_path = os.path.join(
    RESULTS_DIR,
    "epe_validation_probabilities_bank.csv"
)


validation_probability_df.to_csv(
    validation_probability_path,
    index=False
)


print("\nValidation probabilities saved to:")

print(
    validation_probability_path
)


# ============================================================
# 19. PRINT MODEL AUC RANKING
# ============================================================

print("\n")
print("=" * 70)
print("MODEL RANKING BASED ON VALIDATION ROC-AUC")
print("=" * 70)


ranking_df = individual_results_df.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)


print(
    ranking_df[
        [
            "Model",
            "ROC-AUC",
            "F1-Score",
            "Accuracy"
        ]
    ].to_string(index=False)
)


# ============================================================
# 20. SAVE MODEL RANKING
# ============================================================

ranking_path = os.path.join(
    RESULTS_DIR,
    "epe_individual_model_ranking_bank.csv"
)


ranking_df.to_csv(
    ranking_path,
    index=False
)


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("STEP 1 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSix individual models have been trained:")

for name in models:
    print("✓", name)


print("\nFiles created:")

print(
    "1.",
    individual_results_path
)

print(
    "2.",
    validation_probability_path
)

print(
    "3.",
    ranking_path
)


print("\nNext stage:")
print(
    "Multiple hybrid combinations will be evaluated "
    "using the validation probabilities."
)

print("\nDo NOT delete the old EHEM model.")

print("=" * 70)