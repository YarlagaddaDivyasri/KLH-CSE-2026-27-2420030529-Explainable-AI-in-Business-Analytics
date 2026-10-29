import os
import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = r"C:\Users\DELL\OneDrive\Desktop\Business Analytics"

MODEL_DIR = os.path.join(
    BASE_DIR,
    "results",
    "models"
)

OUTPUT_DIR = MODEL_DIR


# ============================================================
# INPUT FILES
# ============================================================

files = {
    "Bank Marketing": "bank_hybrid_results.csv",
    "Online Retail": "online_retail_hybrid_results.csv",
    "Telco Customer Churn": "telco_hybrid_results.csv"
}


# ============================================================
# LOAD RESULTS
# ============================================================

all_results = []

for dataset, filename in files.items():

    file_path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(file_path):

        print("File not found:")
        print(file_path)

        continue

    df = pd.read_csv(file_path)

    print("\nLoaded:", filename)
    print("Columns:", list(df.columns))

    # Add dataset name
    df["Dataset"] = dataset

    all_results.append(df)


# ============================================================
# CHECK RESULTS
# ============================================================

if len(all_results) == 0:

    print("\nNo EHEM result files were found.")
    exit()


# ============================================================
# COMBINE
# ============================================================

final_df = pd.concat(
    all_results,
    ignore_index=True
)


# ============================================================
# KEEP REQUIRED METRICS
# ============================================================

required_columns = [
    "Dataset",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]


# Some scripts may use different column names.
# Rename common alternatives if necessary.

rename_columns = {

    "F1": "F1-Score",
    "F1 Score": "F1-Score",
    "F1_Score": "F1-Score",

    "ROC_AUC": "ROC-AUC",
    "ROC AUC": "ROC-AUC",

    "AUC": "ROC-AUC"
}

final_df = final_df.rename(
    columns=rename_columns
)


# ============================================================
# SELECT AVAILABLE COLUMNS
# ============================================================

available_columns = [
    column
    for column in required_columns
    if column in final_df.columns
]

final_df = final_df[
    available_columns
]


# ============================================================
# ROUND VALUES
# ============================================================

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

for column in metric_columns:

    if column in final_df.columns:

        final_df[column] = final_df[column].round(4)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "final_ehem_comparison.csv"
)

final_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n==============================================")
print("FINAL EHEM MODEL COMPARISON")
print("==============================================")

print(
    final_df.to_string(index=False)
)

print("\nSaved to:")
print(output_file)

print("\n==============================================")
print("COMPARISON COMPLETED")
print("==============================================")