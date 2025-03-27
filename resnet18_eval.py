import os
import re
from collections import defaultdict
from pprint import pprint
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats




def read_results_grouped_by_metric(base_path, source_task):
    result = defaultdict(lambda: defaultdict(dict))

    source_dir = os.path.join(base_path, f"source_{source_task}")

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
            # result["best_train_loss"][source_task][target_task] = numbers[2]
            result["final_train_loss"][source_task][target_task] = numbers[2]
            result["final_test_acc"][source_task][target_task] = numbers[3]
            result["best_test_acc"][source_task][target_task] = numbers[4]

    return result



def read_distances(dist_dir):
    distance_result = torch.zeros(10, 10)

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
            distance_result[source, target] = distance
            distance_result[target, source] = distance

    return distance_result


# Example usage:
# dist_dir = "saved_split_tiny_imagenet/dist"
# distance_tensor = read_distances(dist_dir)

dist_path = "saved_split_tiny_imagenet/dist_pairwise/sotdd_distance.pt"
distance_dict = torch.load(dist_path, map_location='cpu')
distance_tensor = torch.zeros(10, 10)
i = 0
for m in range(10):
    for n in range(m+1, 10):
        if m != n:
            distance_tensor[m, n] = distance_dict[i]
            distance_tensor[n, m] = distance_dict[i]
            i += 1


list_dist = list()
list_acc = list()

SOURCE = 0
base_path = "saved_split_tiny_imagenet/finetune_checkpoints"
results_dict = read_results_grouped_by_metric(base_path, source_task=SOURCE)
accuracies = results_dict["final_train_acc"][SOURCE]

print(accuracies)

for i in range(len(distance_tensor[SOURCE, :])):
    if i == SOURCE:
        continue
    list_dist.append(distance_tensor[SOURCE, i])
    list_acc.append((100 - accuracies[i]) / accuracies[i])


print(list_acc, len(list_acc))
print(list_dist, len(list_dist))


# Ensure all distances are floats
list_dist = [float(d) for d in list_dist]

# list_acc is already float in your example, but just in case:
list_acc = [float(a) for a in list_acc]



# Create DataFrame from collected lists
df = pd.DataFrame({
    "OT Dataset Distance": list_dist,
    "Relative Drop in Test Error (%)": list_acc
})



# Compute Pearson correlation
pearson_corr, p_value = stats.pearsonr(df["OT Dataset Distance"], df["Relative Drop in Test Error (%)"])

# Plot with seaborn
plt.figure(figsize=(7, 7))
sns.set(style="whitegrid")

label=f"$\\rho$: {pearson_corr:.2f}\np-value: {p_value:.2f}"
sns.regplot(
    x="OT Dataset Distance", 
    y="Relative Drop in Test Error (%)", 
    data=df, 
    scatter=True, 
    ci=95, 
    color="c", 
    scatter_kws={"s": 20, "color": "tab:blue"},  # Smaller dots
    label=label
)



# pearson_corr, p_value = stats.pearsonr(list_dist, list_acc)

# df = pd.DataFrame({'OT Dataset Distance': list_dist, 'Accuracy (%)': list_acc})

# plt.figure(figsize=(8, 8))
# sns.set(style="whitegrid")

# plt.scatter(df["OT Dataset Distance"], df["Accuracy (%)"], s=10, color="tab:blue")

# X = np.array(list_dist).reshape(-1, 1)
# y = np.array(list_acc)
# reg = LinearRegression().fit(X, y)

# x_range = np.linspace(min(list_dist), max(list_dist), 500)
# y_pred = reg.predict(x_range.reshape(-1, 1))

# plt.plot(x_range, y_pred, linewidth=1.5, color="tab:blue", label=f"$\\rho$: {pearson_corr:.2f}\np-value: {p_value:.2f}")
plt.legend(loc="upper right", frameon=True)
FONT_SIZE = 15
plt.title(f"Distance vs Adaptation: Tiny-ImageNet", fontsize=FONT_SIZE, fontweight='bold')
plt.xlabel(f"s-OTDD (500,000 projections)", fontsize=FONT_SIZE - 2)
plt.ylabel("Error Drop (%)", fontsize=FONT_SIZE - 2)
plt.grid(False)
saved_dir = 'saved/plots'
os.makedirs(saved_dir, exist_ok=True)
plt.savefig(f'{saved_dir}/aug_s-OTDD.png', dpi=1000)
plt.savefig(f'{saved_dir}/aug_s-OTDD.pdf', dpi=1000)
plt.show()
