import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.linear_model import LinearRegression
from scipy import stats

# Example: Your data
perf_data = [
    {"distance": 0.1094, "performance": -9.02},
    {"distance": 0.1094, "performance": -4.62},
    {"distance": 0.1194, "performance": 49.02},
    {"distance": 0.2054, "performance": -4.87},
    {"distance": 0.2054, "performance": 2.91},
    {"distance": 0.3726, "performance": 9.89},
    {"distance": 0.3726, "performance": 21.82},
    {"distance": 0.3766, "performance": 0.73},
    {"distance": 0.3766, "performance": 7.35},
    {"distance": 0.3774, "performance": 9.31},
    {"distance": 0.3774, "performance": 17.93},
    {"distance": 0.3903, "performance": 8.25},
    {"distance": 0.3903, "performance": 12.73},
    {"distance": 0.3969, "performance": 7.85},
    {"distance": 0.3969, "performance": 16.47},
    {"distance": 0.4254, "performance": 8.29},
    {"distance": 0.4254, "performance": 14.11},
    {"distance": 0.4366, "performance": 11.49},
    {"distance": 0.4369, "performance": 7.2},
    {"distance": 0.5006, "performance": 10.15},
    {"distance": 0.5006, "performance": 12.87},
    {"distance": 0.5037, "performance": 14.69},
    {"distance": 0.5040, "performance": 21.05},
    {"distance": 0.5294, "performance": 8.22},
    {"distance": 0.5294, "performance": 29.64},
    {"distance": 0.5374, "performance": 7.16},
    {"distance": 0.5374, "performance": 46.29},
    {"distance": 0.5437, "performance": 2.47},
    {"distance": 0.5489, "performance": 6.51},
    {"distance": 0.5489, "performance": 7.93},
    {"distance": 0.5554, "performance": 0.55},
    {"distance": 0.5554, "performance": 6.4},
    {"distance": 0.5560, "performance": 22.87},
    {"distance": 0.5566, "performance": 10.8},
    {"distance": 0.5574, "performance": 1.38},
    {"distance": 0.5717, "performance": 11.75},
    {"distance": 0.5717, "performance": 42.51},
    {"distance": 0.5754, "performance": 9.35},
    {"distance": 0.5757, "performance": 23.64},
    {"distance": 0.5926, "performance": 15.45},
    {"distance": 0.5926, "performance": 51.05}
]


df = pd.DataFrame(perf_data)

# Calculate Pearson and Spearman correlation
pearson_corr, p_value_pearson = stats.pearsonr(df["distance"], df["performance"])
spearman_corr, p_value_spearman = stats.spearmanr(df["distance"], df["performance"])

label = (
    f"$\\rho$: {spearman_corr:.2f}  p: {p_value_spearman:.2f}\n"
    f"r: {pearson_corr:.2f}  p: {p_value_pearson:.2f}"
)

# Plotting
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")

sns.regplot(
    x="distance",
    y="performance",
    data=df,
    ci=95,
    color="c",
    scatter_kws={"s": 10, "color": "tab:blue"},
    label=label
)

# Add error bars if present
if "Error" in df.columns:
    plt.errorbar(
        df["distance"],
        df["performance"],
        yerr=df["Error"],
        fmt='o',
        color='gray',
        capsize=1.5,
        capthick=0,
        elinewidth=1,
        markersize=0
    )

# Fit linear regression to extend line beyond data range
X = df["distance"].values.reshape(-1, 1)
y = df["performance"].values
reg = LinearRegression().fit(X, y)

x_range = np.linspace(df["distance"].min(), df["distance"].max(), 500)
y_pred = reg.predict(x_range.reshape(-1, 1))
plt.plot(x_range, y_pred, linestyle="--", color="black", linewidth=1)

# Customize labels and title
FONT_SIZE = 18
plt.title("Distance vs Adaptation: Text Classification", fontsize=FONT_SIZE, fontweight='bold')

method = "sotdd"  # or "otdd"
display_method = method

xlabel = "s-OTDD (10,000 projections)" if method == "sotdd" else "OTDD (Exact)"
plt.xlabel(xlabel, fontsize=FONT_SIZE - 2)
plt.ylabel("Performance Gap (%)", fontsize=FONT_SIZE - 2)

plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.legend(loc="upper right", fontsize=15, frameon=True)
plt.grid(False)

# Save the figure
plt.tight_layout()
plt.savefig(f"text_cls_{display_method}.png", dpi=1000)
plt.savefig(f"text_cls_{display_method}.pdf", dpi=1000)
plt.show()
