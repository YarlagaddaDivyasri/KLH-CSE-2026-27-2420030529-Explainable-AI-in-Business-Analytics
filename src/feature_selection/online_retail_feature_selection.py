import pandas as pd
import numpy as np
import os

from sklearn.feature_selection import (
    mutual_info_classif,
    SelectKBest,
    f_classif
)

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# ONLINE RETAIL - FEATURE SELECTION
# ============================================================

print("=" * 60)
print("ONLINE RETAIL FEATURE SELECTION")
print("=" * 60)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
project_root = os.path.dirname(BASE_DIR)

input_path = os.path.join(
    project_root,
    "results",
    "feature_engineering",
    "online_retail_feature_engineered.csv"
)

output_dir = os.path.join(
    project_root,
    "results",
    "feature_selection"
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

retail = pd.read_csv(input_path)

print("\nDataset loaded successfully!")
print("Original shape:", retail.shape)


# ============================================================
# 3. CREATE BUSINESS TARGET
# ============================================================

# IsHighValue was created during feature engineering.
# It represents transactions whose TotalAmount is
# at or above the 75th percentile.

target = "IsHighValue"

if target not in retail.columns:
    raise ValueError(
        "IsHighValue column was not found."
    )

y = retail[target].astype(int)


print("\nBusiness target:")
print(target)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(
    y.value_counts(normalize=True) * 100
)


# ============================================================
# 4. SELECT CANDIDATE FEATURES
# ============================================================

candidate_features = [
    "Quantity",
    "UnitPrice",
    "Year",
    "Month",
    "Day",
    "Hour",
    "IsReturn",
    "CustomerPurchaseCount",
    "CustomerTotalSpend",
    "CustomerAverageSpend",
    "ProductPurchaseCount"
]


# Keep only columns that actually exist
candidate_features = [
    feature
    for feature in candidate_features
    if feature in retail.columns
]


X = retail[candidate_features].copy()


print("\nCandidate features:")
print(candidate_features)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(
    X.median(numeric_only=True)
)


# ============================================================
# 6. REMOVE CONSTANT FEATURES
# ============================================================

constant_features = [
    column
    for column in X.columns
    if X[column].nunique() <= 1
]

if constant_features:

    print("\nConstant features removed:")
    print(constant_features)

    X = X.drop(
        columns=constant_features
    )


print("\nFeature matrix shape:")
print(X.shape)


# ============================================================
# 7. CORRELATION-BASED SELECTION
# ============================================================

correlation = X.corrwith(y).abs()

correlation = correlation.sort_values(
    ascending=False
)

correlation_df = pd.DataFrame({
    "Feature": correlation.index,
    "Correlation": correlation.values
})


correlation_df.to_csv(
    os.path.join(
        output_dir,
        "online_retail_correlation_features.csv"
    ),
    index=False
)


print("\n" + "=" * 60)
print("TOP FEATURES - CORRELATION")
print("=" * 60)

print(
    correlation_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 8. MUTUAL INFORMATION
# ============================================================

mi_scores = mutual_info_classif(
    X,
    y,
    random_state=42
)

mi_df = pd.DataFrame({
    "Feature": X.columns,
    "MutualInformation": mi_scores
})

mi_df = mi_df.sort_values(
    "MutualInformation",
    ascending=False
)


mi_df.to_csv(
    os.path.join(
        output_dir,
        "online_retail_mutual_information.csv"
    ),
    index=False
)


print("\n" + "=" * 60)
print("TOP FEATURES - MUTUAL INFORMATION")
print("=" * 60)

print(
    mi_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 9. ANOVA F-TEST
# ============================================================

anova_selector = SelectKBest(
    score_func=f_classif,
    k="all"
)

anova_selector.fit(
    X,
    y
)


anova_df = pd.DataFrame({
    "Feature": X.columns,
    "F_Score": anova_selector.scores_,
    "P_Value": anova_selector.pvalues_
})

anova_df = anova_df.sort_values(
    "F_Score",
    ascending=False
)


anova_df.to_csv(
    os.path.join(
        output_dir,
        "online_retail_anova_features.csv"
    ),
    index=False
)


print("\n" + "=" * 60)
print("TOP FEATURES - ANOVA")
print("=" * 60)

print(
    anova_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 10. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(
    X,
    y
)


rf_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

rf_df = rf_df.sort_values(
    "Importance",
    ascending=False
)


rf_df.to_csv(
    os.path.join(
        output_dir,
        "online_retail_random_forest_importance.csv"
    ),
    index=False
)


print("\n" + "=" * 60)
print("TOP FEATURES - RANDOM FOREST")
print("=" * 60)

print(
    rf_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 11. COMBINED RANKING
# ============================================================

results = pd.DataFrame({
    "Feature": X.columns
})


results = results.merge(
    correlation_df,
    on="Feature"
)

results = results.merge(
    mi_df,
    on="Feature"
)

results = results.merge(
    anova_df,
    on="Feature"
)

results = results.merge(
    rf_df,
    on="Feature"
)


# ============================================================
# 12. RANK EACH METHOD
# ============================================================

results["Correlation_Rank"] = (
    results["Correlation"]
    .rank(
        ascending=False,
        method="min"
    )
)


results["MI_Rank"] = (
    results["MutualInformation"]
    .rank(
        ascending=False,
        method="min"
    )
)


results["ANOVA_Rank"] = (
    results["F_Score"]
    .rank(
        ascending=False,
        method="min"
    )
)


results["RF_Rank"] = (
    results["Importance"]
    .rank(
        ascending=False,
        method="min"
    )
)


# ============================================================
# 13. OVERALL RANK
# ============================================================

results["OverallRank"] = (
    results["Correlation_Rank"]
    + results["MI_Rank"]
    + results["ANOVA_Rank"]
    + results["RF_Rank"]
)


results = results.sort_values(
    "OverallRank"
)


# ============================================================
# 14. SAVE COMPARISON
# ============================================================

comparison_path = os.path.join(
    output_dir,
    "online_retail_feature_selection_comparison.csv"
)

results.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 15. SELECT FINAL FEATURES
# ============================================================

final_k = min(
    8,
    len(results)
)

final_features = results.head(
    final_k
)["Feature"].tolist()


print("\n" + "=" * 60)
print("FINAL SELECTED ONLINE RETAIL FEATURES")
print("=" * 60)


for i, feature in enumerate(
    final_features,
    start=1
):

    print(
        f"{i}. {feature}"
    )


# ============================================================
# 16. SAVE SELECTED DATASET
# ============================================================

X_selected = X[
    final_features
].copy()

X_selected[target] = y.values


selected_path = os.path.join(
    output_dir,
    "online_retail_selected_features.csv"
)


X_selected.to_csv(
    selected_path,
    index=False
)


# ============================================================
# 17. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("ONLINE RETAIL FEATURE SELECTION COMPLETED")
print("=" * 60)

print(
    "\nSelected feature count:",
    final_k
)

print(
    "\nComparison saved at:"
)

print(comparison_path)

print(
    "\nSelected dataset saved at:"
)

print(selected_path)