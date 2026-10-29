# ============================================================
# STEP 4
# FINAL TEST EVALUATION
# BANK MARKETING DATASET
# ============================================================
#
# FINAL PROPOSED HYBRID:
#
# XGBoost + HistGradientBoosting
#
# Validation-selected configuration:
#
# XGBoost weight = 0.25
# HGB weight     = 0.75
# Threshold      = 0.29
#
# IMPORTANT:
# The test set is used ONLY in this stage.
# No model/weight/threshold tuning is performed here.
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
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


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
print("STEP 4 - FINAL TEST EVALUATION")
print("BANK MARKETING DATASET")
print("=" * 75)


print("\nProject directory:")
print(BASE_DIR)


# ============================================================
# 2. DATASET PATH
# ============================================================


DATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "feature_selection",
    "bank_selected_features.csv"
)


# ============================================================
# 3. RESULTS DIRECTORY
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
# 4. CHECK DATASET
# ============================================================


if not os.path.exists(DATA_PATH):

    print("\nERROR: Dataset not found.")

    print(
        DATA_PATH
    )

    input("\nPress Enter to exit...")
    exit()


# ============================================================
# 5. LOAD DATASET
# ============================================================


print("\nLoading Bank Marketing dataset...")


df = pd.read_csv(
    DATA_PATH
)


print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================


TARGET_COLUMN = "y"


X = df.drop(
    TARGET_COLUMN,
    axis=1
)


y = df[TARGET_COLUMN]


print(
    "\nFeature shape:",
    X.shape
)


print(
    "Target shape:",
    y.shape
)


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================


missing_values = X.isnull().sum().sum()


print(
    "\nMissing values:",
    missing_values
)


if missing_values > 0:

    X = X.fillna(
        X.median(numeric_only=True)
    )


# ============================================================
# 8. TRAIN / VALIDATION / TEST SPLIT
# ============================================================
#
# IMPORTANT:
#
# This is the SAME split strategy used during the
# validation experiments.
#
# 80% temporary
# 20% test
#
# From temporary:
#
# 80% training
# 20% validation
#
# Final:
#
# 64% training
# 16% validation
# 20% test
#
# ============================================================


print("\nCreating train/validation/test split...")


X_temp, X_test, y_temp, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


X_train, X_val, y_train, y_val = train_test_split(

    X_temp,
    y_temp,

    test_size=0.20,

    random_state=42,

    stratify=y_temp

)


print(
    "\nTraining samples:",
    len(X_train)
)


print(
    "Validation samples:",
    len(X_val)
)


print(
    "Test samples:",
    len(X_test)
)


# ============================================================
# 9. CREATE THE SIX INDIVIDUAL MODELS
# ============================================================


print("\nCreating individual models...")


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
# SVM
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
# 10. MODEL DICTIONARY
# ============================================================


models = {

    "Logistic Regression": lr,

    "Decision Tree": dt,

    "Random Forest": rf,

    "XGBoost": xgb,

    "SVM": svm,

    "HistGradientBoosting": hgb

}


# ============================================================
# 11. TRAIN ALL INDIVIDUAL MODELS
# ============================================================


print("\n")
print("=" * 75)
print("TRAINING MODELS FOR FINAL TEST EVALUATION")
print("=" * 75)


for name, model in models.items():

    print(
        "\nTraining:",
        name
    )


    model.fit(

        X_train,

        y_train

    )


    print(
        name,
        "training completed."
    )


# ============================================================
# 12. EVALUATION FUNCTION
# ============================================================


def calculate_metrics(

    y_true,

    probability,

    threshold=0.50

):


    prediction = (

        probability >= threshold

    ).astype(int)


    accuracy = accuracy_score(

        y_true,

        prediction

    )


    precision = precision_score(

        y_true,

        prediction,

        zero_division=0

    )


    recall = recall_score(

        y_true,

        prediction,

        zero_division=0

    )


    f1 = f1_score(

        y_true,

        prediction,

        zero_division=0

    )


    roc_auc = roc_auc_score(

        y_true,

        probability

    )


    cm = confusion_matrix(

        y_true,

        prediction

    )


    return (

        accuracy,

        precision,

        recall,

        f1,

        roc_auc,

        cm

    )


# ============================================================
# 13. EVALUATE INDIVIDUAL MODELS
# ============================================================


print("\n")
print("=" * 75)
print("FINAL TEST PERFORMANCE - INDIVIDUAL MODELS")
print("=" * 75)


individual_results = []


test_probabilities = {}


for name, model in models.items():


    print(
        "\nEvaluating:",
        name
    )


    probability = model.predict_proba(

        X_test

    )[:, 1]


    test_probabilities[name] = probability


    (

        accuracy,

        precision,

        recall,

        f1,

        roc_auc,

        cm

    ) = calculate_metrics(

        y_test,

        probability,

        threshold=0.50

    )


    individual_results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-Score": f1,

        "ROC-AUC": roc_auc,

        "Threshold": 0.50

    })


    print(

        "Accuracy:",

        f"{accuracy:.6f}"

    )


    print(

        "Precision:",

        f"{precision:.6f}"

    )


    print(

        "Recall:",

        f"{recall:.6f}"

    )


    print(

        "F1-Score:",

        f"{f1:.6f}"

    )


    print(

        "ROC-AUC:",

        f"{roc_auc:.6f}"

    )


    print(
        "Confusion Matrix:"
    )

    print(cm)


# ============================================================
# 14. CREATE INDIVIDUAL RESULTS DATAFRAME
# ============================================================


individual_results_df = pd.DataFrame(

    individual_results

)


# ============================================================
# 15. DISPLAY INDIVIDUAL RESULTS
# ============================================================


print("\n")
print("=" * 75)
print("INDIVIDUAL MODEL TEST RESULTS")
print("=" * 75)


print(

    individual_results_df.to_string(

        index=False

    )

)


# ============================================================
# 16. OPTIMIZED H3 PARAMETERS
# ============================================================
#
# These values were selected ONLY using validation data.
#
# They must NOT be changed based on test performance.
#
# ============================================================


XGB_WEIGHT = 0.25

HGB_WEIGHT = 0.75

FINAL_THRESHOLD = 0.29


# ============================================================
# 17. CREATE FINAL H3 PROBABILITY
# ============================================================


print("\n")
print("=" * 75)
print("FINAL OPTIMIZED H3")
print("=" * 75)


print(
    "\nHybrid:"
)

print(
    "XGBoost + HistGradientBoosting"
)


print(
    "\nXGBoost weight:",
    XGB_WEIGHT
)


print(
    "HistGradientBoosting weight:",
    HGB_WEIGHT
)


print(
    "Validation-selected threshold:",
    FINAL_THRESHOLD
)


h3_test_probability = (

    XGB_WEIGHT
    * test_probabilities["XGBoost"]

    +

    HGB_WEIGHT
    * test_probabilities[
        "HistGradientBoosting"
    ]

)


# ============================================================
# 18. FINAL H3 PREDICTIONS
# ============================================================


h3_test_prediction = (

    h3_test_probability
    >= FINAL_THRESHOLD

).astype(int)


# ============================================================
# 19. FINAL H3 METRICS
# ============================================================


h3_accuracy = accuracy_score(

    y_test,

    h3_test_prediction

)


h3_precision = precision_score(

    y_test,

    h3_test_prediction,

    zero_division=0

)


h3_recall = recall_score(

    y_test,

    h3_test_prediction,

    zero_division=0

)


h3_f1 = f1_score(

    y_test,

    h3_test_prediction,

    zero_division=0

)


h3_roc_auc = roc_auc_score(

    y_test,

    h3_test_probability

)


h3_cm = confusion_matrix(

    y_test,

    h3_test_prediction

)


# ============================================================
# 20. DISPLAY H3 RESULTS
# ============================================================


print("\n")
print("=" * 75)
print("OPTIMIZED H3 - FINAL TEST RESULTS")
print("=" * 75)


print(
    "\nAccuracy:",
    f"{h3_accuracy:.6f}"
)


print(
    "Precision:",
    f"{h3_precision:.6f}"
)


print(
    "Recall:",
    f"{h3_recall:.6f}"
)


print(
    "F1-Score:",
    f"{h3_f1:.6f}"
)


print(
    "ROC-AUC:",
    f"{h3_roc_auc:.6f}"
)


print(
    "\nConfusion Matrix:"
)


print(
    h3_cm
)


# ============================================================
# 21. ADD H3 TO COMPARISON TABLE
# ============================================================


h3_result = {

    "Model":
        "Optimized H3 (XGB + HGB)",

    "Accuracy":
        h3_accuracy,

    "Precision":
        h3_precision,

    "Recall":
        h3_recall,

    "F1-Score":
        h3_f1,

    "ROC-AUC":
        h3_roc_auc,

    "Threshold":
        FINAL_THRESHOLD

}


final_results = individual_results + [

    h3_result

]


final_results_df = pd.DataFrame(

    final_results

)


# ============================================================
# 22. DISPLAY FINAL COMPARISON
# ============================================================


print("\n")
print("=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)


print(

    final_results_df.to_string(

        index=False

    )

)


# ============================================================
# 23. SAVE FINAL COMPARISON
# ============================================================


final_comparison_path = os.path.join(

    RESULTS_DIR,

    "epe_final_test_comparison_bank.csv"

)


final_results_df.to_csv(

    final_comparison_path,

    index=False

)


print("\nFinal comparison saved to:")

print(

    final_comparison_path

)


# ============================================================
# 24. SAVE H3 CONFUSION MATRIX
# ============================================================


h3_cm_path = os.path.join(

    RESULTS_DIR,

    "epe_h3_confusion_matrix_bank.csv"

)


cm_df = pd.DataFrame(

    h3_cm,

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

    h3_cm_path

)


print("\nH3 confusion matrix saved to:")

print(

    h3_cm_path

)


# ============================================================
# 25. SAVE H3 TEST PREDICTIONS
# ============================================================


h3_predictions_path = os.path.join(

    RESULTS_DIR,

    "epe_h3_test_predictions_bank.csv"

)


h3_prediction_df = pd.DataFrame({

    "Actual":
        y_test,

    "H3_Probability":
        h3_test_probability,

    "H3_Prediction":
        h3_test_prediction

})


h3_prediction_df.to_csv(

    h3_predictions_path,

    index=False

)


print("\nH3 test predictions saved to:")

print(

    h3_predictions_path

)


# ============================================================
# 26. COMPARE WITH OLD EHEM
# ============================================================
#
# Existing EHEM result from your previous implementation:
#
# Accuracy  = 0.8884
# Precision = 0.5189
# Recall    = 0.6361
# F1        = 0.5715
# ROC-AUC   = 0.9027
#
# These values are included ONLY as an existing benchmark.
#
# ============================================================


OLD_EHEM_RESULTS = {

    "Model":
        "Old EHEM (LR + RF + XGB)",

    "Accuracy":
        0.8884,

    "Precision":
        0.5189,

    "Recall":
        0.6361,

    "F1-Score":
        0.5715,

    "ROC-AUC":
        0.9027,

    "Threshold":
        0.33

}


comparison_with_ehem = final_results + [

    OLD_EHEM_RESULTS

]


comparison_with_ehem_df = pd.DataFrame(

    comparison_with_ehem

)


# ============================================================
# 27. DISPLAY EHEM COMPARISON
# ============================================================


print("\n")
print("=" * 75)
print("COMPARISON WITH OLD EHEM")
print("=" * 75)


print(

    comparison_with_ehem_df.to_string(

        index=False

    )

)


# ============================================================
# 28. SAVE EHEM COMPARISON
# ============================================================


ehem_comparison_path = os.path.join(

    RESULTS_DIR,

    "epe_vs_old_ehem_bank.csv"

)


comparison_with_ehem_df.to_csv(

    ehem_comparison_path,

    index=False

)


print("\nEPE vs old EHEM comparison saved to:")

print(

    ehem_comparison_path

)


# ============================================================
# 29. CALCULATE IMPROVEMENT OVER OLD EHEM
# ============================================================


print("\n")
print("=" * 75)
print("IMPROVEMENT OF OPTIMIZED H3 OVER OLD EHEM")
print("=" * 75)


print(

    "\nAccuracy difference:",

    f"{h3_accuracy - OLD_EHEM_RESULTS['Accuracy']:.6f}"

)


print(

    "Precision difference:",

    f"{h3_precision - OLD_EHEM_RESULTS['Precision']:.6f}"

)


print(

    "Recall difference:",

    f"{h3_recall - OLD_EHEM_RESULTS['Recall']:.6f}"

)


print(

    "F1 difference:",

    f"{h3_f1 - OLD_EHEM_RESULTS['F1-Score']:.6f}"

)


print(

    "ROC-AUC difference:",

    f"{h3_roc_auc - OLD_EHEM_RESULTS['ROC-AUC']:.6f}"

)


# ============================================================
# 30. FINAL MESSAGE
# ============================================================


print("\n")
print("=" * 75)
print("STEP 4 COMPLETED")
print("=" * 75)


print(
    """
The optimized H3 hybrid has now been evaluated on the
untouched test set.

The final test results are now available.

IMPORTANT:

Do NOT change the XGB/HGB weights or threshold based on
these test results.

The next stage is XAI evaluation of the selected hybrid:

1. SHAP
2. LIME
3. SHAP Fidelity
4. LIME Fidelity
5. SHAP Sparsity
6. LIME Sparsity
7. SHAP-LIME consistency

Only after the final model has been selected should XAI
be applied.
"""
)


print("=" * 75)