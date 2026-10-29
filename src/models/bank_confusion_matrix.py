import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# CONFUSION MATRIX VISUALIZATION
# ============================================================

input_dir = r"results\models"
output_dir = r"results\models"

models = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost"
]

for model in models:

    # Load confusion matrix
    file_path = os.path.join(
        input_dir,
        f"bank_{model}_confusion_matrix.csv"
    )

    cm = pd.read_csv(
        file_path,
        index_col=0
    )

    # Create figure
    plt.figure(figsize=(6, 5))

    plt.imshow(cm, interpolation="nearest")

    plt.title(
        f"Bank Marketing - {model.replace('_', ' ').title()}"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    # Axis labels
    plt.xticks(
        range(2),
        ["Class 0", "Class 1"]
    )

    plt.yticks(
        range(2),
        ["Class 0", "Class 1"]
    )

    # Add values inside the matrix
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm.iloc[i, j],
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    # Save image
    output_file = os.path.join(
        output_dir,
        f"bank_{model}_confusion_matrix.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.show()

    print("Saved:", output_file)


print("\nConfusion matrix visualization completed.")