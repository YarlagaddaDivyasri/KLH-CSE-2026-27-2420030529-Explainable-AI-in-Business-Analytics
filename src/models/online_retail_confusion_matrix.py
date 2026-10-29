import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
results_path = "results/models"
output_path = "results/models"

# Model names
models = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost"
]

# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

for model in models:

    # Confusion matrix file
    file_path = f"{results_path}/online_retail_{model}_confusion_matrix.csv"

    # Load confusion matrix
    cm = pd.read_csv(file_path, index_col=0)

    # Plot
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"]
    )

    plt.title(f"{model.replace('_', ' ').title()} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    # Save image
    output_file = f"{output_path}/online_retail_{model}_confusion_matrix.png"

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved: {output_file}")

print("\nAll Online Retail confusion matrices generated successfully.")