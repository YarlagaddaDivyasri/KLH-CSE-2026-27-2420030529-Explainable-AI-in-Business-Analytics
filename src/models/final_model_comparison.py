import pandas as pd
import os


# --------------------------------------------------
# 1. Load model result files
# --------------------------------------------------

bank_file = "results/models/bank_model_results.csv"
online_retail_file = "results/models/online_retail_model_results.csv"
telco_file = "results/models/telco_model_results.csv"


bank = pd.read_csv(bank_file)
online_retail = pd.read_csv(online_retail_file)
telco = pd.read_csv(telco_file)


# --------------------------------------------------
# 2. Add Dataset column
# --------------------------------------------------

bank["Dataset"] = "Bank Marketing"
online_retail["Dataset"] = "Online Retail"
telco["Dataset"] = "Telco Customer Churn"


# --------------------------------------------------
# 3. Combine all results
# --------------------------------------------------

comparison = pd.concat(
    [bank, online_retail, telco],
    ignore_index=True
)


# --------------------------------------------------
# 4. Arrange columns
# --------------------------------------------------

comparison = comparison[
    [
        "Dataset",
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]
]


# --------------------------------------------------
# 5. Round values
# --------------------------------------------------

comparison[
    ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
] = comparison[
    ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
].round(4)


# --------------------------------------------------
# 6. Display comparison
# --------------------------------------------------

print("\nFinal Model Comparison")
print("=" * 80)

print(comparison.to_string(index=False))


# --------------------------------------------------
# 7. Save result
# --------------------------------------------------

output_file = "results/models/final_model_comparison.csv"

os.makedirs("results/models", exist_ok=True)

comparison.to_csv(
    output_file,
    index=False
)

print("\nSaved:", output_file)

print("\nFinal model comparison completed successfully.")