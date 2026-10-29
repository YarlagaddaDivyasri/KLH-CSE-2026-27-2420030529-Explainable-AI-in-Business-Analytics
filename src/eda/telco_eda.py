import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================
# TELCO CUSTOMER CHURN - EDA
# ============================================

print("=" * 60)
print("TELCO CUSTOMER CHURN DATASET - EDA")
print("=" * 60)

# --------------------------------------------
# 1. Paths
# --------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
project_root = os.path.dirname(BASE_DIR)

input_path = os.path.join(
    project_root,
    "results",
    "preprocessing",
    "telco_preprocessed.csv"
)

output_dir = os.path.join(
    project_root,
    "results",
    "eda",
    "telco"
)

os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------
# 2. Load dataset
# --------------------------------------------

telco = pd.read_csv(input_path)

print("\nDataset loaded successfully!")
print("Shape:", telco.shape)


# --------------------------------------------
# 3. Basic information
# --------------------------------------------

print("\nColumns:")
print(telco.columns.tolist())

print("\nData types:")
print(telco.dtypes)

print("\nMissing values:")
print(telco.isnull().sum())

print("\nDuplicate records:")
print(telco.duplicated().sum())


# --------------------------------------------
# 4. Statistical summary
# --------------------------------------------

print("\nStatistical Summary:")
print(telco.describe())


# --------------------------------------------
# 5. Target distribution
# --------------------------------------------

print("\nChurn Distribution:")
print(telco["Churn"].value_counts())

print("\nChurn Percentage:")
print(
    telco["Churn"].value_counts(normalize=True) * 100
)


plt.figure(figsize=(7, 5))

sns.countplot(
    data=telco,
    x="Churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "churn_distribution.png"
    )
)

plt.close()


# --------------------------------------------
# 6. Numerical columns
# --------------------------------------------

numerical_columns = telco.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)


# --------------------------------------------
# 7. Numerical distributions
# --------------------------------------------

for column in numerical_columns:

    if column == "Churn":
        continue

    plt.figure(figsize=(7, 5))

    sns.histplot(
        data=telco,
        x=column,
        kde=True
    )

    plt.title(
        f"Distribution of {column}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{column}_distribution.png"
        )
    )

    plt.close()


# --------------------------------------------
# 8. Correlation matrix
# --------------------------------------------

if len(numerical_columns) > 1:

    correlation = telco[
        numerical_columns
    ].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f"
    )

    plt.title(
        "Telco Customer Churn Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "correlation_matrix.png"
        )
    )

    plt.close()


# --------------------------------------------
# 9. Tenure vs churn
# --------------------------------------------

if "tenure" in telco.columns:

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=telco,
        x="Churn",
        y="tenure"
    )

    plt.title(
        "Tenure Distribution by Churn"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "tenure_vs_churn.png"
        )
    )

    plt.close()


# --------------------------------------------
# 10. Monthly charges vs churn
# --------------------------------------------

if "MonthlyCharges" in telco.columns:

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=telco,
        x="Churn",
        y="MonthlyCharges"
    )

    plt.title(
        "Monthly Charges by Churn"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "monthly_charges_vs_churn.png"
        )
    )

    plt.close()


# --------------------------------------------
# 11. Total charges vs churn
# --------------------------------------------

if "TotalCharges" in telco.columns:

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=telco,
        x="Churn",
        y="TotalCharges"
    )

    plt.title(
        "Total Charges by Churn"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "total_charges_vs_churn.png"
        )
    )

    plt.close()


# --------------------------------------------
# 12. Important categorical features
# --------------------------------------------

categorical_features = [
    column for column in telco.columns
    if column.startswith((
        "Contract_",
        "InternetService_",
        "PaymentMethod_",
        "TechSupport_",
        "OnlineSecurity_"
    ))
]


# --------------------------------------------
# 13. Categorical feature analysis
# --------------------------------------------

for column in categorical_features[:5]:

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=telco,
        x=column,
        hue="Churn"
    )

    plt.title(
        f"{column} vs Churn"
    )

    plt.xticks(rotation=30)

    plt.tight_layout()

    safe_name = column.replace(
        " ",
        "_"
    )

    plt.savefig(
        os.path.join(
            output_dir,
            f"{safe_name}_vs_churn.png"
        )
    )

    plt.close()


# --------------------------------------------
# 14. Churn rate by tenure group
# --------------------------------------------

if "tenure" in telco.columns:

    telco["TenureGroup"] = pd.cut(
        telco["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ],
        include_lowest=True
    )

    churn_by_tenure = (
        telco.groupby(
            "TenureGroup",
            observed=True
        )["Churn"]
        .mean()
        * 100
    )

    print("\nChurn rate by tenure group:")
    print(churn_by_tenure)


# --------------------------------------------
# 15. Print observations
# --------------------------------------------

print("\n" + "=" * 60)
print("TELCO EDA COMPLETED")
print("=" * 60)

print("\nEDA results saved at:")
print(output_dir)