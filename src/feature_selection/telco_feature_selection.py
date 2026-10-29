import pandas as pd
import os

from sklearn.feature_selection import (
    mutual_info_classif,
    SelectKBest,
    f_classif,
    RFE
)

from sklearn.ensemble import RandomForestClassifier


# ============================================
# TELCO CUSTOMER CHURN - FEATURE SELECTION
# ============================================

print("=" * 60)
print("TELCO CUSTOMER CHURN FEATURE SELECTION")
print("=" * 60)


# --------------------------------------------
# 1. Paths
# --------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
project_root = os.path.dirname(BASE_DIR)

input_path = os.path.join(
    project_root,
    "results",
    "feature_engineering",
    "telco_feature_engineered.csv"
)

output_dir = os.path.join(
    project_root,
    "results",
    "feature_selection"
)

os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------
# 2. Load dataset
# --------------------------------------------

telco = pd.read_csv(input_path)

print("\nDataset loaded successfully!")
print("Shape:", telco.shape)


# --------------------------------------------
# 3. Separate target
# --------------------------------------------

target = "Churn"

X = telco.drop(columns=[target])
y = telco[target]


# --------------------------------------------
# 4. Keep numeric features
# --------------------------------------------

X_encoded = X.select_dtypes(
    include=["int64", "float64"]
)

print("\nNumeric feature shape:")
print(X_encoded.shape)


# --------------------------------------------
# 5. Remove constant columns
# --------------------------------------------

constant_columns = [
    column
    for column in X_encoded.columns
    if X_encoded[column].nunique() <= 1
]

if constant_columns:

    X_encoded = X_encoded.drop(
        columns=constant_columns
    )

    print("\nConstant columns removed:")
    print(constant_columns)


# --------------------------------------------
# 6. Correlation-based selection
# --------------------------------------------

correlation = X_encoded.corrwith(y).abs()

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
        "telco_correlation_features.csv"
    ),
    index=False
)

print("\nTop 10 features by correlation:")
print(correlation_df.head(10))


# --------------------------------------------
# 7. Mutual Information
# --------------------------------------------

mi_scores = mutual_info_classif(
    X_encoded,
    y,
    random_state=42
)

mi_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "MutualInformation": mi_scores
})

mi_df = mi_df.sort_values(
    "MutualInformation",
    ascending=False
)

mi_df.to_csv(
    os.path.join(
        output_dir,
        "telco_mutual_information.csv"
    ),
    index=False
)

print("\nTop 10 features by Mutual Information:")
print(mi_df.head(10))


# --------------------------------------------
# 8. ANOVA F-test
# --------------------------------------------

selector = SelectKBest(
    score_func=f_classif,
    k="all"
)

selector.fit(
    X_encoded,
    y
)

anova_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "F_Score": selector.scores_,
    "P_Value": selector.pvalues_
})

anova_df = anova_df.sort_values(
    "F_Score",
    ascending=False
)

anova_df.to_csv(
    os.path.join(
        output_dir,
        "telco_anova_features.csv"
    ),
    index=False
)

print("\nTop 10 features by ANOVA:")
print(anova_df.head(10))


# --------------------------------------------
# 9. Random Forest importance
# --------------------------------------------

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(
    X_encoded,
    y
)

rf_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": rf.feature_importances_
})

rf_df = rf_df.sort_values(
    "Importance",
    ascending=False
)

rf_df.to_csv(
    os.path.join(
        output_dir,
        "telco_random_forest_importance.csv"
    ),
    index=False
)

print("\nTop 10 features by Random Forest:")
print(rf_df.head(10))


# --------------------------------------------
# 10. RFE
# --------------------------------------------

n_features = min(
    10,
    X_encoded.shape[1]
)

rfe = RFE(
    estimator=RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),
    n_features_to_select=n_features
)

rfe.fit(
    X_encoded,
    y
)

rfe_df = pd.DataFrame({
    "Feature": X_encoded.columns,
    "RFE_Rank": rfe.ranking_
})

rfe_df = rfe_df.sort_values(
    "RFE_Rank"
)

rfe_df.to_csv(
    os.path.join(
        output_dir,
        "telco_rfe_features.csv"
    ),
    index=False
)

print("\nTop features selected by RFE:")
print(
    rfe_df[
        rfe_df["RFE_Rank"] == 1
    ]
)


# --------------------------------------------
# 11. Combined ranking
# --------------------------------------------

results = pd.DataFrame({
    "Feature": X_encoded.columns
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

results = results.merge(
    rfe_df,
    on="Feature"
)


# --------------------------------------------
# 12. Ranking
# --------------------------------------------

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


results["OverallRank"] = (
    results["Correlation_Rank"]
    + results["MI_Rank"]
    + results["ANOVA_Rank"]
    + results["RF_Rank"]
    + results["RFE_Rank"]
)


results = results.sort_values(
    "OverallRank"
)


# --------------------------------------------
# 13. Save comparison
# --------------------------------------------

comparison_path = os.path.join(
    output_dir,
    "telco_feature_selection_comparison.csv"
)

results.to_csv(
    comparison_path,
    index=False
)


# --------------------------------------------
# 14. Select final features
# --------------------------------------------

final_k = min(
    15,
    len(results)
)

final_features = results.head(
    final_k
)["Feature"].tolist()


print("\n" + "=" * 60)
print("FINAL SELECTED TELCO FEATURES")
print("=" * 60)

for i, feature in enumerate(
    final_features,
    start=1
):

    print(
        f"{i}. {feature}"
    )


# --------------------------------------------
# 15. Save selected dataset
# --------------------------------------------

X_selected = X_encoded[
    final_features
].copy()

X_selected[target] = y.values

selected_path = os.path.join(
    output_dir,
    "telco_selected_features.csv"
)

X_selected.to_csv(
    selected_path,
    index=False
)


# --------------------------------------------
# 16. Completion
# --------------------------------------------

print("\n" + "=" * 60)
print("TELCO FEATURE SELECTION COMPLETED")
print("=" * 60)

print("\nSelected feature count:", final_k)

print("\nComparison saved at:")
print(comparison_path)

print("\nSelected dataset saved at:")
print(selected_path)