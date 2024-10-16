import json
import numpy as np 
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats


display_method = "sOTDD"
baseline_result_path = "saved/text_cls_spp/text_baseline2.txt"
adapt_result_path = "saved/text_cls_spp/text_adapt.txt"
text_dist_path = f"saved/text_cls_new2/dist/{display_method}_text_dist3.json"


# read text distance
with open(text_dist_path, "r") as file:
    text_dist = json.load(file)

# read adapt result
adapt_acc = {}
with open(adapt_result_path, 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        source_dataset = parts[0].split(': ')[1]
        target_dataset = parts[1].split(': ')[1]
        accuracy = float(parts[2].split(': ')[1])

        if source_dataset not in adapt_acc:
            adapt_acc[source_dataset] = {}
        adapt_acc[source_dataset][target_dataset] = accuracy


# read baseline result
baseline_acc = {}
with open(baseline_result_path, 'r') as file:
    for line in file:
        parts = line.strip().split(': ')
        # print(parts)
        source_dataset = parts[1].split(', ')[0]
        accuracy = float(parts[3])
        baseline_acc[source_dataset] = accuracy

print(baseline_acc)


mean = 0.0
std_dev = 0.001

perf_list = list()
dist_list = list()
DATASET_NAME = list(baseline_acc.keys())
print(DATASET_NAME)
for i in range(len(DATASET_NAME)):
    for j in range(len(DATASET_NAME)):
        source = DATASET_NAME[i]
        target = DATASET_NAME[j]
        if source == target:
            continue
        if source == "AmazonReviewPolarity" or target == "AmazonReviewPolarity":
            continue

        # gaussian_numbers = torch.normal(mean=mean, std=std_dev, size=(1,))
        perf = ((adapt_acc[source][target]) - (baseline_acc[target]))
        dist = text_dist[source][target]

        # if dist > 0.1:
        perf_list.append(perf)
        dist_list.append(dist)

list_X = np.array(dist_list).reshape(-1, 1)
list_y = np.array(perf_list)
model = LinearRegression().fit(list_X, list_y)
list_y_pred = model.predict(list_X)
plt.figure(figsize=(10, 8))
# sns.regplot(x=x, y=y, ci=95, scatter_kws={'s': 100}, line_kws={'color': 'blue'})
plt.scatter(dist_list, perf_list, s=50, color='blue', label='Data points')
plt.plot(dist_list, list_y_pred, color='red', linewidth=2, label='Fitted line')

rho, p_value = stats.pearsonr(dist_list, perf_list)
print(f"rho: {rho}, p value: {p_value}")

FONT_SIZE = 25
plt.title(f'{display_method} $\\rho={rho:.3f}, p={p_value:.3f}$', fontsize=FONT_SIZE)  # Increase title size
plt.xlabel(f'{display_method}', fontsize=FONT_SIZE)  # Increase x-axis label size
plt.ylabel('Accuracy', fontsize=FONT_SIZE)  # Increase y-axis label size


plt.legend()
plt.savefig(f'text_cls_{display_method}.pdf')


