# ============================================================
# STEP 3
# OPTIMIZATION OF H3 HYBRID
#
# H3 = XGBoost + HistGradientBoosting
#
# This program:
# 1. Loads validation probabilities
# 2. Tests different ensemble weights
# 3. Tests different classification thresholds
# 4. Finds the best validation F1
# 5. Finds the best validation ROC-AUC
# 6. Saves all experiments
#
# IMPORTANT:
# The test set is NOT used here.
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# 1. PROJECT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


print("=" * 75)
print("STEP 3 - H3 HYBRID OPTIMIZATION")
print("XGBoost + HistGradientBoosting")
print("=" * 75)


print("\nProject directory:")
print(BASE_DIR)


# ============================================================
# 2. INPUT FILE
# ============================================================

PROBABILITY_PATH = os.path.join(
    BASE_DIR,
    "results",
    "models",
    "epe_validation_probabilities_bank.csv"
)


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

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
# 4. CHECK INPUT
# ============================================================

if not os.path.exists(PROBABILITY_PATH):

    print("\nERROR:")
    print("Validation probability file not found.")

    print("\nExpected:")
    print(PROBABILITY_PATH)

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# 5. LOAD VALIDATION PROBABILITIES
# ============================================================

print("\nLoading validation probabilities...")

df = pd.read_csv(
    PROBABILITY_PATH
)


print(
    "Validation data shape:",
    df.shape
)


# ============================================================
# 6. GET REQUIRED DATA
# ============================================================

y_val = df["Actual"].values


xgb_probability = (
    df["XGBoost"].values
)


hgb_probability = (
    df["HistGradientBoosting"].values
)


# ============================================================
# 7. INITIALIZE RESULTS
# ============================================================

optimization_results = []


# ============================================================
# 8. WEIGHT SEARCH
# ============================================================
#
# XGB weight:
# 0.10 to 0.90
#
# HGB weight:
# 1 - XGB weight
#
# Step = 0.05
#
# ============================================================


xgb_weights = np.arange(
    0.10,
    0.91,
    0.05
)


# ============================================================
# 9. THRESHOLD SEARCH
# ============================================================
#
# We search from 0.20 to 0.70.
#
# This is important because the Bank Marketing dataset
# is imbalanced.
#
# ============================================================


thresholds = np.arange(
    0.20,
    0.71,
    0.01
)


print("\n")
print("=" * 75)
print("SEARCHING WEIGHTS AND THRESHOLDS")
print("=" * 75)


# ============================================================
# 10. GRID SEARCH
# ============================================================


for xgb_weight in xgb_weights:

    hgb_weight = 1.0 - xgb_weight


    # --------------------------------------------------------
    # Combine probabilities
    # --------------------------------------------------------

    hybrid_probability = (

        xgb_weight
        * xgb_probability

        +

        hgb_weight
        * hgb_probability

    )


    # --------------------------------------------------------
    # ROC-AUC does not depend on threshold
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_val,
        hybrid_probability
    )


    # --------------------------------------------------------
    # Test different thresholds
    # --------------------------------------------------------

    for threshold in thresholds:

        hybrid_prediction = (
            hybrid_probability >= threshold
        ).astype(int)


        accuracy = accuracy_score(
            y_val,
            hybrid_prediction
        )


        precision = precision_score(
            y_val,
            hybrid_prediction,
            zero_division=0
        )


        recall = recall_score(
            y_val,
            hybrid_prediction,
            zero_division=0
        )


        f1 = f1_score(
            y_val,
            hybrid_prediction,
            zero_division=0
        )


        optimization_results.append({

            "XGB Weight":
                xgb_weight,

            "HGB Weight":
                hgb_weight,

            "Threshold":
                threshold,

            "Accuracy":
                accuracy,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1-Score":
                f1,

            "ROC-AUC":
                roc_auc

        })


# ============================================================
# 11. CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    optimization_results
)


print("\nTotal experiments:")
print(
    len(results_df)
)


# ============================================================
# 12. BEST F1 RESULT
# ============================================================

best_f1 = (
    results_df
    .sort_values(
        by="F1-Score",
        ascending=False
    )
    .iloc[0]
)


# ============================================================
# 13. BEST ROC-AUC RESULT
# ============================================================

best_auc = (
    results_df
    .sort_values(
        by="ROC-AUC",
        ascending=False
    )
    .iloc[0]
)


# ============================================================
# 14. DISPLAY BEST F1
# ============================================================

print("\n")
print("=" * 75)
print("BEST H3 CONFIGURATION BASED ON F1-SCORE")
print("=" * 75)


print(
    "\nXGBoost Weight:",
    f"{best_f1['XGB Weight']:.2f}"
)


print(
    "HistGradientBoosting Weight:",
    f"{best_f1['HGB Weight']:.2f}"
)


print(
    "Threshold:",
    f"{best_f1['Threshold']:.2f}"
)


print(
    "Accuracy:",
    f"{best_f1['Accuracy']:.6f}"
)


print(
    "Precision:",
    f"{best_f1['Precision']:.6f}"
)


print(
    "Recall:",
    f"{best_f1['Recall']:.6f}"
)


print(
    "F1-Score:",
    f"{best_f1['F1-Score']:.6f}"
)


print(
    "ROC-AUC:",
    f"{best_f1['ROC-AUC']:.6f}"
)


# ============================================================
# 15. DISPLAY BEST ROC-AUC
# ============================================================

print("\n")
print("=" * 75)
print("BEST H3 CONFIGURATION BASED ON ROC-AUC")
print("=" * 75)


print(
    "\nXGBoost Weight:",
    f"{best_auc['XGB Weight']:.2f}"
)


print(
    "HistGradientBoosting Weight:",
    f"{best_auc['HGB Weight']:.2f}"
)


print(
    "ROC-AUC:",
    f"{best_auc['ROC-AUC']:.6f}"
)


# ============================================================
# 16. TOP 10 CONFIGURATIONS BY F1
# ============================================================

top_10_f1 = (
    results_df
    .sort_values(
        by="F1-Score",
        ascending=False
    )
    .head(10)
)


print("\n")
print("=" * 75)
print("TOP 10 H3 CONFIGURATIONS BY F1-SCORE")
print("=" * 75)


print(
    top_10_f1.to_string(
        index=False
    )
)


# ============================================================
# 17. SAVE ALL OPTIMIZATION RESULTS
# ============================================================

all_results_path = os.path.join(
    RESULTS_DIR,
    "epe_h3_weight_threshold_search_bank.csv"
)


results_df.to_csv(
    all_results_path,
    index=False
)


print("\nAll optimization results saved to:")

print(
    all_results_path
)


# ============================================================
# 18. SAVE TOP 10
# ============================================================

top_10_path = os.path.join(
    RESULTS_DIR,
    "epe_h3_top10_configurations_bank.csv"
)


top_10_f1.to_csv(
    top_10_path,
    index=False
)


print("\nTop 10 configurations saved to:")

print(
    top_10_path
)


# ============================================================
# 19. SAVE BEST CONFIGURATION
# ============================================================

best_config_path = os.path.join(
    RESULTS_DIR,
    "epe_h3_best_configuration_bank.csv"
)


best_f1.to_frame().T.to_csv(
    best_config_path,
    index=False
)


print("\nBest configuration saved to:")

print(
    best_config_path
)


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 75)
print("STEP 3 COMPLETED")
print("=" * 75)


print(
    """
H3 optimization is complete.

The best validation configuration has been identified.

Next stage:

1. Retrain the selected models on the training data.
2. Generate test probabilities.
3. Apply the selected validation weights.
4. Apply the selected validation threshold.
5. Evaluate ONCE on the untouched test set.
6. Compare against:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - XGBoost
   - SVM
   - HistGradientBoosting
   - Old EHEM
   - Optimized H3
"""
)


print("\nDo NOT use the test set for further tuning.")

print("=" * 75)