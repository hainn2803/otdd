import os
import re
from collections import defaultdict
from pprint import pprint
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats


def read_results_grouped_by_metric(base_path, source_task=0):
    source_dir = os.path.join(base_path, f"source_{source_task}")
    
    # Initialize result dict with default nested dicts
    result = defaultdict(lambda: defaultdict(dict))

    for filename in os.listdir(source_dir):
        if filename.endswith(".txt") and filename.startswith(f"results_source_{source_task}_"):
            # Extract target task number from filename
            match = re.search(r"target_(\d+)", filename)
            if not match:
                continue
            target_task = int(match.group(1))

            file_path = os.path.join(source_dir, filename)
            with open(file_path, 'r') as f:
                content = f.read()

            # Extract all numeric values
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)
            numbers = list(map(float, numbers))

            # Assign values to each metric
            result["final_train_acc"][source_task][target_task] = numbers[0]
            result["best_train_acc"][source_task][target_task] = numbers[1]
            result["best_train_loss"][source_task][target_task] = numbers[2]
            result["final_train_loss"][source_task][target_task] = numbers[3]
            result["final_test_acc"][source_task][target_task] = numbers[4]
            result["best_test_acc"][source_task][target_task] = numbers[5]

    return result



def read_distances(dist_dir):
    distance_result = defaultdict(dict)

    for filename in os.listdir(dist_dir):
        if filename.endswith(".pt") and filename.startswith("sourcce_"):
            filepath = os.path.join(dist_dir, filename)

            # Extract source and target numbers from filename
            match = re.search(r"sourcce_(\d+)_target_(\d+)\.pt", filename)
            if not match:
                continue
            source = int(match.group(1))
            target = int(match.group(2))

            # Load the tensor (distance value)
            try:
                tensor = torch.load(filepath)
                distance = tensor.item()  # Convert single-element tensor to float
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")
                continue

            # Store both directions since distance is symmetric
            distance_result[source][target] = distance
            distance_result[target][source] = distance

    return distance_result


# Example usage:
dist_dir = "saved_split_tiny_imagenet/dist"
distance_dict = read_distances(dist_dir)

base_path = "saved_split_tiny_imagenet/finetune_checkpoints"
results_dict = read_results_grouped_by_metric(base_path, source_task=0)


# distances = distance_dict[0]
# accuracies = results_dict["final_test_acc"][0]

# print(distances)
# print(accuracies)

list_dist = list()
list_acc = list()

for source_id in distance_dict:
    for target_id in distance_dict[source]:
        


list_dist = [distances[k] for k in distances]
list_acc = [accuracies[k] for k in distances]

pearson_corr, p_value = stats.pearsonr(list_dist, list_acc)

df = pd.DataFrame({'OT Dataset Distance': list_dist, 'Accuracy (%)': list_acc})

plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")

plt.scatter(df["OT Dataset Distance"], df["Accuracy (%)"], s=10, color="tab:blue")

X = np.array(list_dist).reshape(-1, 1)
y = np.array(list_acc)
reg = LinearRegression().fit(X, y)

x_range = np.linspace(min(list_dist), max(list_dist), 500)
y_pred = reg.predict(x_range.reshape(-1, 1))

plt.plot(x_range, y_pred, linewidth=1.5, color="tab:blue", label=f"$\\rho$: {pearson_corr:.2f}\np-value: {p_value:.2f}")

plt.legend(loc="upper right", frameon=True)

FONT_SIZE = 20
plt.title(f"Dist vs Adapt: Tiny ImageNet", fontsize=FONT_SIZE, fontweight='bold')
plt.xlabel(f"s-OTDD (100,000 projections)", fontsize=FONT_SIZE - 2)
plt.ylabel("Accuracy (%)", fontsize=FONT_SIZE - 2)
plt.grid(False)

# ==== Save ====
saved_dir = 'saved/plots'
os.makedirs(saved_dir, exist_ok=True)
plt.savefig(f'{saved_dir}/aug_s-OTDD.png', dpi=1000)
plt.savefig(f'{saved_dir}/aug_s-OTDD.pdf', dpi=1000)
plt.show()
