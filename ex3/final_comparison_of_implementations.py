import re
import numpy as np
import matplotlib.pyplot as plt

# ========================== EXTRACT RESULTS ==========================

# Extract Accuracy from TPOT and PyCaret
def extract_tpot_pycaret_results(file_path):
    accuracy = None  

    try:
        with open(file_path, "r") as file:
            for line in file:
                acc_match = re.search(r"Accuracy:\s+(\d+\.\d+)", line)  
                if acc_match:
                    accuracy = float(acc_match.group(1)) * 100  # Convert to percentage
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Setting accuracy to 0.")
        accuracy = 0

    return accuracy

# Extract Best Accuracy from Custom
def extract_custom_results(file_path):
    accuracy = None  

    try:
        with open(file_path, "r") as file:
            log_data = file.readlines()

        for line in log_data:
            acc_match = re.search(r"Best score: (\d+\.\d+)", line)  # Match best score (accuracy)
            if acc_match:
                accuracy = float(acc_match.group(1)) * 100  # Convert to percentage
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Setting accuracy to 0.")
        accuracy = 0

    return accuracy

# ========================== PROCESS MULTIPLE DATASETS ==========================

# Dataset names for better readability
datasets = ["Sick", "Congressional Voting", "Waveform", "Wine"]

# Store accuracy results
tpot_all_accuracy = []
pycaret_all_accuracy = []
custom_all_accuracy = []

for i, dataset in enumerate(datasets, start=1):
    print(f"Processing {dataset} dataset...")

    # File paths based on naming convention
    tpot_file = f"tpot_ds_{i}_results.txt"
    pycaret_file = f"pycaret_ds_{i}_results.txt"
    custom_file = f"custom_sim_ann_for_ds_{i}.log"

    # Extract accuracy results
    tpot_accuracy = extract_tpot_pycaret_results(tpot_file)
    pycaret_accuracy = extract_tpot_pycaret_results(pycaret_file)
    custom_accuracy = extract_custom_results(custom_file)

    # Store for later plotting
    tpot_all_accuracy.append(tpot_accuracy)
    pycaret_all_accuracy.append(pycaret_accuracy)
    custom_all_accuracy.append(custom_accuracy)

# ========================== GENERATE IMPROVED BAR PLOT ==========================

# Extract values for plotting
x = np.arange(len(datasets))  # Dataset indices
width = 0.25  # Bar width

fig, ax = plt.subplots(figsize=(12, 6))

# Plot bars for each implementation
ax.bar(x - width, tpot_all_accuracy, width, label="TPOT", color="blue", edgecolor='black')
ax.bar(x, pycaret_all_accuracy, width, label="PyCaret", color="green", edgecolor='black')
ax.bar(x + width, custom_all_accuracy, width, label="Custom", color="red", edgecolor='black')

# Labels and Titles
ax.set_xlabel("Datasets", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy Comparison Across Datasets for the Implementations", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10, rotation=15)

# Grid for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Move the legend outside the plot (upper right corner)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)

# Save and Show Plot
plt.savefig("accuracy_comparison_of_implementations.png", dpi=300, bbox_inches="tight")
plt.show()
