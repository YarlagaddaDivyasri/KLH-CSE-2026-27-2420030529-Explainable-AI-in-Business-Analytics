import pandas as pd
import matplotlib.pyplot as plt
import os


# --------------------------------------------------
# 1. Load Final Comparison Data
# --------------------------------------------------

input_file = "results/models/final_model_comparison.csv"

df = pd.read_csv(input_file)

print("Final comparison data loaded successfully.")
print("Shape:", df.shape)


# --------------------------------------------------
# 2. Create Output Folder
# --------------------------------------------------

output_folder = "results/models"

os.makedirs(output_folder, exist_ok=True)


# --------------------------------------------------
# 3. Metrics to Compare
# --------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]


# --------------------------------------------------
# 4. Create One Graph for Each Metric
# --------------------------------------------------

for metric in metrics:

    plt.figure(figsize=(12, 6))

    for dataset in df["Dataset"].unique():

        dataset_data = df[df["Dataset"] == dataset]

        plt.plot(
            dataset_data["Model"],
            dataset_data[metric],
            marker="o",
            label=dataset
        )

    plt.title(
        f"{metric} Comparison Across Datasets"
    )

    plt.xlabel("Machine Learning Model")

    plt.ylabel(metric)

    plt.ylim(0, 1)

    plt.xticks(rotation=15)

    plt.legend()

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = (
        f"{output_folder}/"
        f"final_{metric.lower().replace('-', '_')}_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_file)


print(
    "\nFinal performance comparison graphs "
    "created successfully."
)