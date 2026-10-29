import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================
# ONLINE RETAIL DATASET - EDA
# ============================================

print("=" * 60)
print("ONLINE RETAIL DATASET - EDA")
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
    "online_retail_preprocessed.csv"
)

output_dir = os.path.join(
    project_root,
    "results",
    "eda",
    "online_retail"
)

os.makedirs(output_dir, exist_ok=True)


# --------------------------------------------
# 2. Load dataset
# --------------------------------------------

retail = pd.read_csv(input_path)

print("\nDataset loaded successfully!")
print("Shape:", retail.shape)


# --------------------------------------------
# 3. Basic information
# --------------------------------------------

print("\nColumns:")
print(retail.columns.tolist())

print("\nData types:")
print(retail.dtypes)

print("\nMissing values:")
print(retail.isnull().sum())

print("\nDuplicate records:")
print(retail.duplicated().sum())


# --------------------------------------------
# 4. Statistical summary
# --------------------------------------------

print("\nStatistical Summary:")
print(retail.describe())


# --------------------------------------------
# 5. Quantity distribution
# --------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=retail,
    x="Quantity",
    bins=50
)

plt.title("Quantity Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "quantity_distribution.png"
    )
)

plt.close()


# --------------------------------------------
# 6. Unit Price distribution
# --------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=retail,
    x="UnitPrice",
    bins=50
)

plt.title("Unit Price Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "unit_price_distribution.png"
    )
)

plt.close()


# --------------------------------------------
# 7. Total Amount distribution
# --------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    data=retail,
    x="TotalAmount",
    bins=50
)

plt.title("Total Amount Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "total_amount_distribution.png"
    )
)

plt.close()


# --------------------------------------------
# 8. Top countries
# --------------------------------------------

top_countries = (
    retail["Country"]
    .value_counts()
    .head(10)
)

print("\nTop 10 Countries:")
print(top_countries)


plt.figure(figsize=(10, 6))

top_countries.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Countries by Number of Transactions")
plt.xlabel("Number of Transactions")
plt.ylabel("Country")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "top_countries.png"
    )
)

plt.close()


# --------------------------------------------
# 9. Monthly sales
# --------------------------------------------

if "Year" in retail.columns and "Month" in retail.columns:

    monthly_sales = (
        retail
        .groupby(["Year", "Month"])["TotalAmount"]
        .sum()
        .reset_index()
    )

    monthly_sales["Period"] = (
        monthly_sales["Year"].astype(str)
        + "-"
        + monthly_sales["Month"].astype(str)
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=monthly_sales,
        x="Period",
        y="TotalAmount",
        marker="o"
    )

    plt.title("Monthly Sales Trend")
    plt.xlabel("Year-Month")
    plt.ylabel("Total Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "monthly_sales_trend.png"
        )
    )

    plt.close()


# --------------------------------------------
# 10. Correlation analysis
# --------------------------------------------

numerical_columns = retail.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

if len(numerical_columns) > 1:

    correlation = retail[
        numerical_columns
    ].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f"
    )

    plt.title("Online Retail Correlation Matrix")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "correlation_matrix.png"
        )
    )

    plt.close()


# --------------------------------------------
# 11. Top products
# --------------------------------------------

if "Description" in retail.columns:

    top_products = (
        retail["Description"]
        .value_counts()
        .head(10)
    )

    print("\nTop 10 Products:")
    print(top_products)

    plt.figure(figsize=(10, 6))

    top_products.sort_values().plot(
        kind="barh"
    )

    plt.title("Top 10 Most Frequently Purchased Products")
    plt.xlabel("Number of Transactions")
    plt.ylabel("Product")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "top_products.png"
        )
    )

    plt.close()


# --------------------------------------------
# 12. Customer transaction analysis
# --------------------------------------------

if "CustomerID" in retail.columns:

    customer_transactions = (
        retail["CustomerID"]
        .value_counts()
        .head(10)
    )

    print("\nTop 10 Customers by Transaction Count:")
    print(customer_transactions)


# --------------------------------------------
# 13. Print observations
# --------------------------------------------

print("\n" + "=" * 60)
print("ONLINE RETAIL EDA COMPLETED")
print("=" * 60)

print("\nEDA results saved at:")
print(output_dir)