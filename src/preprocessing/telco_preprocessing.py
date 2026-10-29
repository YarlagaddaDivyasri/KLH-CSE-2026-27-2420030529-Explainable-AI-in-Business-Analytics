import pandas as pd
import numpy as np
import os

# ============================================
# TELCO CUSTOMER CHURN - PREPROCESSING
# ============================================

print("=" * 60)
print("TELCO CUSTOMER CHURN DATASET PREPROCESSING")
print("=" * 60)

# --------------------------------------------
# 1. Dataset path
# --------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

dataset_path = os.path.join(
    BASE_DIR,
    "datasets",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("\nDataset path:")
print(dataset_path)


# --------------------------------------------
# 2. Load dataset
# --------------------------------------------

telco = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")
print("Original shape:", telco.shape)


# --------------------------------------------
# 3. Basic information
# --------------------------------------------

print("\nFirst 5 records:")
print(telco.head())

print("\nColumn names:")
print(telco.columns.tolist())

print("\nData types:")
print(telco.dtypes)


# --------------------------------------------
# 4. Remove customer ID
# --------------------------------------------

if "customerID" in telco.columns:
    telco = telco.drop(columns=["customerID"])

    print("\nCustomerID removed because it is an identifier.")


# --------------------------------------------
# 5. Convert TotalCharges to numeric
# --------------------------------------------

if "TotalCharges" in telco.columns:

    telco["TotalCharges"] = pd.to_numeric(
        telco["TotalCharges"],
        errors="coerce"
    )


# --------------------------------------------
# 6. Check duplicates
# --------------------------------------------

duplicate_count = telco.duplicated().sum()

print("\nDuplicate records:", duplicate_count)

if duplicate_count > 0:
    telco = telco.drop_duplicates()

print("Shape after removing duplicates:", telco.shape)


# --------------------------------------------
# 7. Check missing values
# --------------------------------------------

print("\nMissing values before preprocessing:")
print(telco.isnull().sum())


# --------------------------------------------
# 8. Handle missing values
# --------------------------------------------

numerical_columns = telco.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = telco.select_dtypes(
    include=["object", "string"]
).columns.tolist()


for column in numerical_columns:

    telco[column] = telco[column].fillna(
        telco[column].median()
    )


for column in categorical_columns:

    if telco[column].isnull().sum() > 0:

        telco[column] = telco[column].fillna(
            telco[column].mode()[0]
        )


# --------------------------------------------
# 9. Encode target variable
# --------------------------------------------

if "Churn" in telco.columns:

    telco["Churn"] = telco["Churn"].map({
        "Yes": 1,
        "No": 0
    })


# --------------------------------------------
# 10. Convert categorical variables
# --------------------------------------------

categorical_columns = telco.select_dtypes(
    include=["object", "string"]
).columns.tolist()

categorical_columns = [
    column for column in categorical_columns
    if column != "Churn"
]

telco = pd.get_dummies(
    telco,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)


# --------------------------------------------
# 11. Final missing-value check
# --------------------------------------------

remaining_missing = telco.isnull().sum().sum()

print("\nRemaining missing values:", remaining_missing)


# --------------------------------------------
# 12. Final information
# --------------------------------------------

print("\nFinal dataset shape:", telco.shape)

print("\nFinal columns:")
print(telco.columns.tolist())


# --------------------------------------------
# 13. Save processed dataset
# --------------------------------------------

project_root = os.path.dirname(BASE_DIR)

output_dir = os.path.join(
    project_root,
    "results",
    "preprocessing"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

output_path = os.path.join(
    output_dir,
    "telco_preprocessed.csv"
)

telco.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 60)
print("TELCO PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nProcessed dataset saved at:")
print(output_path)