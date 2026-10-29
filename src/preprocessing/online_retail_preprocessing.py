import pandas as pd
import numpy as np
import os

# ============================================
# ONLINE RETAIL DATASET - PREPROCESSING
# ============================================

print("=" * 60)
print("ONLINE RETAIL DATASET PREPROCESSING")
print("=" * 60)

# --------------------------------------------
# 1. Dataset path
# --------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

dataset_path = os.path.join(
    BASE_DIR,
    "datasets",
    "Online Retail.xlsx"
)

print("\nDataset path:")
print(dataset_path)


# --------------------------------------------
# 2. Load dataset
# --------------------------------------------


retail = pd.read_excel(
    dataset_path,
    engine="openpyxl"
)

print("\nDataset loaded successfully!")
print("Original shape:", retail.shape)


# --------------------------------------------
# 3. Display information
# --------------------------------------------

print("\nFirst 5 records:")
print(retail.head())

print("\nColumn names:")
print(retail.columns.tolist())

print("\nData types:")
print(retail.dtypes)


# --------------------------------------------
# 4. Check duplicates
# --------------------------------------------

duplicate_count = retail.duplicated().sum()

print("\nDuplicate records:", duplicate_count)

if duplicate_count > 0:
    retail = retail.drop_duplicates()

print("Shape after removing duplicates:", retail.shape)


# --------------------------------------------
# 5. Check missing values
# --------------------------------------------

print("\nMissing values:")
print(retail.isnull().sum())


# --------------------------------------------
# 6. Handle missing CustomerID
# --------------------------------------------

if "CustomerID" in retail.columns:

    missing_customer = retail["CustomerID"].isnull().sum()

    print("\nMissing CustomerID:", missing_customer)

    retail = retail.dropna(
        subset=["CustomerID"]
    )


# --------------------------------------------
# 7. Handle missing Description
# --------------------------------------------

if "Description" in retail.columns:

    missing_description = retail["Description"].isnull().sum()

    print("Missing Description:", missing_description)

    retail["Description"] = retail["Description"].fillna(
        "Unknown"
    )


# --------------------------------------------
# 8. Remove invalid quantities
# --------------------------------------------

if "Quantity" in retail.columns:

    invalid_quantity = (retail["Quantity"] <= 0).sum()

    print("\nInvalid quantity records:", invalid_quantity)

    retail = retail[
        retail["Quantity"] > 0
    ]


# --------------------------------------------
# 9. Remove invalid prices
# --------------------------------------------

if "UnitPrice" in retail.columns:

    invalid_price = (retail["UnitPrice"] <= 0).sum()

    print("Invalid price records:", invalid_price)

    retail = retail[
        retail["UnitPrice"] > 0
    ]


# --------------------------------------------
# 10. Convert InvoiceDate
# --------------------------------------------

if "InvoiceDate" in retail.columns:

    retail["InvoiceDate"] = pd.to_datetime(
        retail["InvoiceDate"],
        errors="coerce"
    )


# --------------------------------------------
# 11. Feature Engineering
# --------------------------------------------

if "Quantity" in retail.columns and "UnitPrice" in retail.columns:

    retail["TotalAmount"] = (
        retail["Quantity"] *
        retail["UnitPrice"]
    )


# --------------------------------------------
# 12. Extract date features
# --------------------------------------------

if "InvoiceDate" in retail.columns:

    retail["Year"] = retail["InvoiceDate"].dt.year
    retail["Month"] = retail["InvoiceDate"].dt.month
    retail["Day"] = retail["InvoiceDate"].dt.day
    retail["Hour"] = retail["InvoiceDate"].dt.hour


# --------------------------------------------
# 13. Final missing-value check
# --------------------------------------------

print("\nMissing values after preprocessing:")

print(retail.isnull().sum())

print(
    "\nTotal remaining missing values:",
    retail.isnull().sum().sum()
)


# --------------------------------------------
# 14. Final dataset information
# --------------------------------------------

print("\nFinal dataset shape:", retail.shape)

print("\nFinal data types:")
print(retail.dtypes)


# --------------------------------------------
# 15. Save processed dataset
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
    "online_retail_preprocessed.csv"
)

retail.to_csv(
    output_path,
    index=False
)


print("\n" + "=" * 60)
print("ONLINE RETAIL PREPROCESSING COMPLETED")
print("=" * 60)

print("\nProcessed dataset saved at:")
print(output_path)