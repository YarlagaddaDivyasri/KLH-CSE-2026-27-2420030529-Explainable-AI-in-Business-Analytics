import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


print("=" * 60)
print("BANK MARKETING DATASET - EDA")
print("=" * 60)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

project_root = os.path.dirname(BASE_DIR)

input_path = os.path.join(
    project_root,
    "results",
    "preprocessing",
    "bank_preprocessed.csv"
)

project_root = os.path.dirname(BASE_DIR)

output_dir = os.path.join(
    project_root,
    "results",
    "eda",
    "bank"
)

os.makedirs(output_dir, exist_ok=True)



bank = pd.read_csv(input_path)

print("\nDataset loaded successfully!")
print("Shape:", bank.shape)



print("\nColumns:")
print(bank.columns.tolist())

print("\nData types:")
print(bank.dtypes)

print("\nMissing values:")
print(bank.isnull().sum())

print("\nDuplicate records:")
print(bank.duplicated().sum())



print("\nStatistical Summary:")
print(bank.describe())




print("\nTarget Distribution:")
print(bank["y"].value_counts())

print("\nTarget Percentage:")
print(bank["y"].value_counts(normalize=True) * 100)


plt.figure(figsize=(7, 5))

sns.countplot(
    data=bank,
    x="y"
)

plt.title("Bank Marketing Target Distribution")
plt.xlabel("Subscription (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "target_distribution.png"
    )
)

plt.close()



numerical_columns = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

for column in numerical_columns:

    plt.figure(figsize=(7, 5))

    sns.histplot(
        data=bank,
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
# 7. Correlation matrix
# --------------------------------------------

plt.figure(figsize=(10, 8))

correlation = bank[
    numerical_columns + ["y"]
].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)

plt.title("Bank Marketing Correlation Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "correlation_matrix.png"
    )
)

plt.close()


# --------------------------------------------
# 8. Age vs target
# --------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=bank,
    x="y",
    y="age"
)

plt.title("Age Distribution by Subscription")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "age_vs_target.png"
    )
)

plt.close()



plt.figure(figsize=(7, 5))

sns.boxplot(
    data=bank,
    x="y",
    y="balance"
)

plt.title("Balance Distribution by Subscription")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "balance_vs_target.png"
    )
)

plt.close()



plt.figure(figsize=(10, 6))

sns.countplot(
    data=bank,
    y="job",
    order=bank["job"].value_counts().index
)

plt.title("Customer Distribution by Job")

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "job_distribution.png"
    )
)

plt.close()




plt.figure(figsize=(8, 5))

sns.countplot(
    data=bank,
    x="education",
    order=bank["education"].value_counts().index
)

plt.title("Customer Distribution by Education")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "education_distribution.png"
    )
)

plt.close()



print("\n" + "=" * 60)
print("EDA COMPLETED")
print("=" * 60)

print("\nTarget counts:")
print(bank["y"].value_counts())

print("\nTop 5 job categories:")
print(bank["job"].value_counts().head())

print("\nAverage numerical values by target:")
print(
    bank.groupby("y")[numerical_columns].mean()
)

print("\nEDA results saved at:")
print(output_dir)