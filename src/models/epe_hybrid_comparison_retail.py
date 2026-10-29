import os
import itertools
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


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

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "models"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 80)
print("ONLINE RETAIL - HYBRID MODEL COMPARISON")
print("=" * 80)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "RepeatCustomer"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )

y = df[TARGET].astype(int)

X = df.drop(
    columns=[TARGET]
).copy()


# ============================================================
# 4. REMOVE IDENTIFIERS IF PRESENT
# ============================================================

# These columns should not be used directly as predictors.
identifier_columns = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "CustomerID"
]

columns_to_remove = [
    col for col in identifier_columns
    if col in X.columns
]

if columns_to_remove:

    print(
        "\nRemoving identifier columns:"
    )

    print(columns_to_remove)

    X = X.drop(
        columns=columns_to_remove
    )


# ============================================================
# 5. DISPLAY TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

print(
    y.value_counts()
)

print("\nTarget percentage:")

print(
    (y.value_counts(normalize=True) * 100)
    .round(4)
)


# ============================================================
# 6. DETECT NUMERIC / CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 7. PREPROCESSOR
# ============================================================

transformers = []

if numeric_features:

    transformers.append(
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    )

if categorical_features:

    transformers.append(
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ============================================================
# 8. TRAIN / VALIDATION / TEST SPLIT
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


print("\n" + "=" * 80)
print("DATA SPLIT")
print("=" * 80)

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
# 9. MODEL DEFINITIONS
# ============================================================

models = {

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    "Logistic Regression": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),


    # --------------------------------------------------------
    # Decision Tree
    # --------------------------------------------------------

    "Decision Tree": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42
            )
        )
    ]),


    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    "Random Forest": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]),


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    "XGBoost": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ]),


    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------

    "SVM": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=42,
                    max_iter=5000
                ),
                method="sigmoid",
                cv=3
            )
        )
    ]),


    # --------------------------------------------------------
    # HistGradientBoosting
    # --------------------------------------------------------

    "HistGradientBoosting": Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=31,
                random_state=42
            )
        )
    ])
}


# ============================================================
# 10. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    y_true,
    probabilities,
    threshold=0.50
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {

        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities
        )
    }


# ============================================================
# 11. TRAIN BASELINE MODELS
# ============================================================

trained_models = {}

train_probabilities = {}
val_probabilities = {}
test_probabilities = {}

baseline_results = []


print("\n" + "=" * 80)
print("TRAINING BASELINE MODELS")
print("=" * 80)


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    trained_models[name] = model


    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    train_probability = (
        model.predict_proba(X_train)[:, 1]
    )

    validation_probability = (
        model.predict_proba(X_val)[:, 1]
    )

    test_probability = (
        model.predict_proba(X_test)[:, 1]
    )


    train_probabilities[name] = (
        train_probability
    )

    val_probabilities[name] = (
        validation_probability
    )

    test_probabilities[name] = (
        test_probability
    )


    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    metrics = evaluate_model(
        y_test,
        test_probability,
        threshold=0.50
    )

    metrics["Model"] = name

    baseline_results.append(
        metrics
    )


    print(
        f"Accuracy : {metrics['Accuracy']:.6f}"
    )

    print(
        f"Precision: {metrics['Precision']:.6f}"
    )

    print(
        f"Recall   : {metrics['Recall']:.6f}"
    )

    print(
        f"F1       : {metrics['F1']:.6f}"
    )

    print(
        f"ROC-AUC  : {metrics['ROC-AUC']:.6f}"
    )


# ============================================================
# 12. SAVE BASELINE RESULTS
# ============================================================

baseline_df = pd.DataFrame(
    baseline_results
)

baseline_df = baseline_df[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC"
    ]
]

baseline_path = os.path.join(
    RESULTS_DIR,
    "online_retail_baseline_models.csv"
)

baseline_df.to_csv(
    baseline_path,
    index=False
)


print(
    f"\nBaseline results saved:\n{baseline_path}"
)


# ============================================================
# 13. SAVE TRAINED MODELS
# ============================================================

MODEL_DIR = os.path.join(
    RESULTS_DIR,
    "online_retail_models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


for name, model in trained_models.items():

    safe_name = (
        name
        .lower()
        .replace(" ", "_")
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"{safe_name}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )


# ============================================================
# 14. SAVE PROBABILITIES
# ============================================================

probability_df = pd.DataFrame({
    "y_test": y_test.reset_index(drop=True),

    "Logistic Regression":
        test_probabilities[
            "Logistic Regression"
        ],

    "Decision Tree":
        test_probabilities[
            "Decision Tree"
        ],

    "Random Forest":
        test_probabilities[
            "Random Forest"
        ],

    "XGBoost":
        test_probabilities[
            "XGBoost"
        ],

    "SVM":
        test_probabilities[
            "SVM"
        ],

    "HistGradientBoosting":
        test_probabilities[
            "HistGradientBoosting"
        ]
})

probability_path = os.path.join(
    RESULTS_DIR,
    "online_retail_test_probabilities.csv"
)

probability_df.to_csv(
    probability_path,
    index=False
)


# ============================================================
# 15. HYBRID COMBINATIONS
# ============================================================

hybrid_combinations = [

    (
        "RF + XGB",
        [
            "Random Forest",
            "XGBoost"
        ]
    ),

    (
        "RF + HGB",
        [
            "Random Forest",
            "HistGradientBoosting"
        ]
    ),

    (
        "XGB + HGB",
        [
            "XGBoost",
            "HistGradientBoosting"
        ]
    ),

    (
        "SVM + XGB",
        [
            "SVM",
            "XGBoost"
        ]
    ),

    (
        "SVM + HGB",
        [
            "SVM",
            "HistGradientBoosting"
        ]
    ),

    (
        "RF + SVM + XGB",
        [
            "Random Forest",
            "SVM",
            "XGBoost"
        ]
    ),

    (
        "RF + XGB + HGB",
        [
            "Random Forest",
            "XGBoost",
            "HistGradientBoosting"
        ]
    ),

    (
        "SVM + XGB + HGB",
        [
            "SVM",
            "XGBoost",
            "HistGradientBoosting"
        ]
    )
]


# ============================================================
# 16. HYBRID COMPARISON
# ============================================================

hybrid_results = []


print("\n" + "=" * 80)
print("TESTING HYBRID COMBINATIONS")
print("=" * 80)


for hybrid_name, members in hybrid_combinations:

    print(
        f"\nTesting {hybrid_name}..."
    )


    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    validation_matrix = np.column_stack([
        val_probabilities[member]
        for member in members
    ])


    # Equal-weight hybrid
    validation_probability = np.mean(
        validation_matrix,
        axis=1
    )


    # --------------------------------------------------------
    # Test probabilities
    # --------------------------------------------------------

    test_matrix = np.column_stack([
        test_probabilities[member]
        for member in members
    ])


    test_probability = np.mean(
        test_matrix,
        axis=1
    )


    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    validation_metrics = evaluate_model(
        y_val,
        validation_probability,
        threshold=0.50
    )


    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    test_metrics = evaluate_model(
        y_test,
        test_probability,
        threshold=0.50
    )


    # --------------------------------------------------------
    # Prediction diversity
    # --------------------------------------------------------

    diversity_values = []


    for model_a, model_b in itertools.combinations(
        members,
        2
    ):

        correlation = np.corrcoef(
            val_probabilities[model_a],
            val_probabilities[model_b]
        )[0, 1]

        diversity_values.append(
            1 - correlation
        )


    if diversity_values:

        diversity = np.mean(
            diversity_values
        )

    else:

        diversity = 0.0


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    hybrid_results.append({

        "Hybrid":
            hybrid_name,

        "Validation_Accuracy":
            validation_metrics["Accuracy"],

        "Validation_Precision":
            validation_metrics["Precision"],

        "Validation_Recall":
            validation_metrics["Recall"],

        "Validation_F1":
            validation_metrics["F1"],

        "Validation_ROC_AUC":
            validation_metrics["ROC-AUC"],

        "Test_Accuracy":
            test_metrics["Accuracy"],

        "Test_Precision":
            test_metrics["Precision"],

        "Test_Recall":
            test_metrics["Recall"],

        "Test_F1":
            test_metrics["F1"],

        "Test_ROC_AUC":
            test_metrics["ROC-AUC"],

        "Prediction_Diversity":
            diversity
    })


    print(
        f"Validation F1  : "
        f"{validation_metrics['F1']:.6f}"
    )

    print(
        f"Validation AUC : "
        f"{validation_metrics['ROC-AUC']:.6f}"
    )

    print(
        f"Test F1        : "
        f"{test_metrics['F1']:.6f}"
    )

    print(
        f"Test AUC       : "
        f"{test_metrics['ROC-AUC']:.6f}"
    )

    print(
        f"Diversity      : "
        f"{diversity:.6f}"
    )


# ============================================================
# 17. CREATE HYBRID DATAFRAME
# ============================================================

hybrid_df = pd.DataFrame(
    hybrid_results
)


# ============================================================
# 18. SORT USING VALIDATION PERFORMANCE
# ============================================================

hybrid_df = hybrid_df.sort_values(
    by=[
        "Validation_ROC_AUC",
        "Validation_F1"
    ],
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 19. SAVE HYBRID RESULTS
# ============================================================

hybrid_path = os.path.join(
    RESULTS_DIR,
    "online_retail_hybrid_comparison.csv"
)

hybrid_df.to_csv(
    hybrid_path,
    index=False
)


# ============================================================
# 20. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("HYBRID COMPARISON RESULTS")
print("=" * 80)

print(
    hybrid_df[
        [
            "Hybrid",
            "Validation_F1",
            "Validation_ROC_AUC",
            "Test_F1",
            "Test_ROC_AUC",
            "Prediction_Diversity"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 21. SELECT BEST HYBRID
# ============================================================

best_hybrid = hybrid_df.iloc[0]


print("\n" + "=" * 80)
print("BEST HYBRID")
print("=" * 80)

print(
    "\nHybrid:",
    best_hybrid["Hybrid"]
)

print(
    "Validation Accuracy:",
    round(
        best_hybrid["Validation_Accuracy"],
        6
    )
)

print(
    "Validation Precision:",
    round(
        best_hybrid["Validation_Precision"],
        6
    )
)

print(
    "Validation Recall:",
    round(
        best_hybrid["Validation_Recall"],
        6
    )
)

print(
    "Validation F1:",
    round(
        best_hybrid["Validation_F1"],
        6
    )
)

print(
    "Validation ROC-AUC:",
    round(
        best_hybrid["Validation_ROC_AUC"],
        6
    )
)

print(
    "Prediction Diversity:",
    round(
        best_hybrid["Prediction_Diversity"],
        6
    )
)


# ============================================================
# 22. SAVE BEST HYBRID
# ============================================================

best_hybrid_df = pd.DataFrame([
    best_hybrid
])

best_hybrid_path = os.path.join(
    RESULTS_DIR,
    "online_retail_best_hybrid.csv"
)

best_hybrid_df.to_csv(
    best_hybrid_path,
    index=False
)


# ============================================================
# 23. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("IMPLEMENTATION COMPLETED")
print("=" * 80)

print(
    "\nSaved files:"
)

print(
    "1.",
    baseline_path
)

print(
    "2.",
    hybrid_path
)

print(
    "3.",
    best_hybrid_path
)

print(
    "4.",
    probability_path
)

print(
    "5.",
    MODEL_DIR
)

print("\nNext step will be:")
print(
    "Validation-based hybrid weight + threshold optimization."
)