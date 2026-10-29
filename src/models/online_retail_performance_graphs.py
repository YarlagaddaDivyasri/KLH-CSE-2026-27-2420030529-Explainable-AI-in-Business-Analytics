import pandas as pd
import matplotlib.pyplot as plt
import os

# Load model results
input_file = "results/models/online_retail_model_results.csv"
output_folder = "results/models"

os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(input_file)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

for metric in metrics:

    plt.figure(figsize=(8, 5))

    plt.bar(df["Model"], df[metric])

    plt.title(f"Online Retail - {metric} Comparison")
    plt.xlabel("Model")
    plt.ylabel(metric)

    plt.ylim(0, 1)

    plt.xticks(rotation=15)

    plt.tight_layout()

    # Convert metric name to filename format
    metric_name = metric.lower().replace("-", "_")

    output_file = (
        f"{output_folder}/online_retail_{metric_name}_comparison.png"
    )

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved: {output_file}")

print("\nAll Online Retail performance graphs generated successfully.")