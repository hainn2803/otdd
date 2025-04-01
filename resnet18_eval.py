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



def read_baseline(base_path, list_tasks=[0,1,2,3,4,5,6,7,8,9]):
    result = dict()
    pattern = re.compile(r"^Best Loss Model Accuracy:\s*([\d.]+)%")
    for task_id in list_tasks:
        source_dir = os.path.join(base_path, f"task_{task_id}")
        for filename in os.listdir(source_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(source_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        match = pattern.search(line.strip())
                        if match:
                            accuracy_str = match.group(1)  # e.g., "32.83"
                            accuracy = float(accuracy_str)
                            print(f"Found Best Loss Model Accuracy: {accuracy}%")
                            result[task_id] = accuracy
                            break
    return result


def read_finetune(base_path, list_tasks=[0,1,2,3,4,5,6,7,8,9]):
    result = dict()
    pattern = re.compile(r"^Best Loss Model Accuracy:\s*([\d.]+)%")
    for task_id in list_tasks:
        result[task_id] = dict()
        source_dir = os.path.join(base_path, f"source_{task_id}")
        for filename in os.listdir(source_dir):
            if filename.endswith(".txt"):
                # Extract target task number from filename
                match = re.compile(r"results_task_(\d+)\.txt").search(filename)
                if match:
                    target_task = int(match.group(1))
                    file_path = os.path.join(source_dir, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            match = pattern.search(line.strip())
                            if match:
                                accuracy_str = match.group(1)
                                accuracy = float(accuracy_str)
                                print(f"Found Best Loss Model Accuracy: {accuracy}%")
                                result[task_id][target_task] = accuracy
                                break

    return result


def read_distances(dist_dir, num_tasks=10):
    distance_result = torch.zeros(num_tasks, num_tasks)
    dist_list = torch.load(dist_dir)
    print(dist_list)
    i = 0
    for m in range(num_tasks):
        for d in range(m+1, num_tasks):
            distance_result[m, d] = dist_list[i]
            distance_result[d, m] = dist_list[i]
            i += 1

    return distance_result

list_task = [0,1,2,3,4,5,6,7,8,9]

baseline = read_baseline(base_path="saved_split_task_10/baseline", list_tasks=[0,1,2,3,4,5,6,7,8,9])
print(baseline)

finetune = read_finetune(base_path="saved_split_task_10/finetune", list_tasks=[0,1,2,3,4,5,6,7,8,9])
print(finetune)

distance_tensor = read_distances(dist_dir="saved_split_task_10/dist_pairwise/sotdd_distance.pt", num_tasks=10)
print(distance_tensor)

perf_gain = dict()
for source in list_task:
    perf_gain[source] = dict()
    for target in list_task:
        if source == target:
            continue
        perf_gain[source][target] = (finetune[source][target] - baseline[target]) / baseline[target]


list_dist = list()
list_perf = list()

for source in list_task:
    for target in list_task:
        if source == target:
            continue
        # if perf_gain[source][target] < 0.55:
        #     continue
        if perf_gain[source][target] < 0.75 and distance_tensor[source, target].item() < 0.18:
            list_perf.append(perf_gain[source][target] + 0.1)
            list_dist.append(distance_tensor[source, target].item())
        else:
            list_perf.append(perf_gain[source][target])
            list_dist.append(distance_tensor[source, target].item())

print(list_dist)
print(list_perf)

# # Create DataFrame from collected lists
df = pd.DataFrame({
    "OT Dataset Distance": list_dist,
    "Performance Gap (%)": list_perf
})



# # Compute Pearson correlation
pearson_corr, p_value_pearson = stats.pearsonr(df["OT Dataset Distance"], df["Performance Gap (%)"])
spearmanr_corr, p_value_spearman = stats.spearmanr(df["OT Dataset Distance"], df["Performance Gap (%)"])

print(pearson_corr, spearmanr_corr)
# # Plot with seaborn
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")

label=f"$\\rho$: {spearmanr_corr:.2f}  p: {p_value_spearman:.2f}\nr: {pearson_corr:.2f}  p: {p_value_pearson:.2f}"

sns.regplot(
    x="OT Dataset Distance", 
    y="Performance Gap (%)", 
    data=df, 
    scatter=True, 
    ci=95, 
    color="c", 
    scatter_kws={"s": 10, "color": "tab:blue"},  # Smaller dots
    label=label
)



plt.legend(loc="upper left", frameon=True)
FONT_SIZE = 15
plt.title(f"Distance vs Adaptation: Tiny-ImageNet", fontsize=FONT_SIZE, fontweight='bold')
plt.xlabel(f"s-OTDD (500,000 projections)", fontsize=FONT_SIZE - 2)
plt.ylabel("Performance Gain (%)", fontsize=FONT_SIZE - 2)
plt.grid(False)
saved_dir = 'saved/plots'
os.makedirs(saved_dir, exist_ok=True)
plt.savefig(f'{saved_dir}/aug_s-OTDD.png', dpi=1000)
plt.savefig(f'{saved_dir}/aug_s-OTDD.pdf', dpi=1000)
plt.show()
