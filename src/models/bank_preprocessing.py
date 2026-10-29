import pandas as pd
import numpy as np
import os

# ============================================
# BANK MARKETING DATASET - PREPROCESSING
# ============================================

print("=" * 60)
print("BANK MARKETING DATASET PREPROCESSING")
print("=" * 60)

# --------------------------------------------
# 1. Dataset path
# --------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

dataset_path = os.path.join(
    BASE_DIR,
    "datasets",
    "bank-full.csv"
)

print("\nDataset path:")
print(dataset_path)


# --------------------------------------------
# 2. Load dataset
# --------------------------------------------

bank = pd.read_csv(
    dataset_path,
    sep=";"
)

print("\nDataset loaded successfully!")

print("Original shape:", bank.shape)


# --------------------------------------------
# 3. Display basic information
# --------------------------------------------

print("\nFirst 5 records:")
print(bank.head())

print("\nColumn names:")
print(bank.columns.tolist())

print("\nData types:")
print(bank.dtypes)


# --------------------------------------------
# 4. Check duplicate records
# --------------------------------------------

duplicate_count = bank.duplicated().sum()

print("\nDuplicate records:", duplicate_count)

if duplicate_count > 0:
    bank = bank.drop_duplicates()

print("Shape after removing duplicates:", bank.shape)


# --------------------------------------------
# 5. Check missing values
# --------------------------------------------

print("\nMissing values before preprocessing:")
print(bank.isnull().sum())


# --------------------------------------------
# 6. Convert 'unknown' into missing values
# --------------------------------------------

bank = bank.replace("unknown", np.nan)

print("\nMissing values after treating 'unknown':")
print(bank.isnull().sum())


# --------------------------------------------
# 7. Identify numerical columns
# --------------------------------------------

numerical_columns = bank.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)


# --------------------------------------------
# 8. Identify categorical columns
# --------------------------------------------

categorical_columns = bank.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)


# --------------------------------------------
# 9. Handle numerical missing values
# --------------------------------------------

for column in numerical_columns:

    bank[column] = bank[column].fillna(
        bank[column].median()
    )


# --------------------------------------------
# 10. Handle categorical missing values
# --------------------------------------------

for column in categorical_columns:

    bank[column] = bank[column].fillna(
        bank[column].mode()[0]
    )


# --------------------------------------------
# 11. Encode target variable
# --------------------------------------------

if "y" in bank.columns:

    bank["y"] = bank["y"].map({
        "yes": 1,
        "no": 0
    })


# --------------------------------------------
# 12. Final missing value check
# --------------------------------------------

remaining_missing = bank.isnull().sum().sum()

print("\nRemaining missing values:", remaining_missing)


# --------------------------------------------
# 13. Final dataset information
# --------------------------------------------

print("\nFinal dataset shape:", bank.shape)

print("\nFinal data types:")
print(bank.dtypes)


# --------------------------------------------
# 14. Save processed dataset
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
    "bank_preprocessed.csv"
)

bank.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nProcessed dataset saved at:")
print(output_path)