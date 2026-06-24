# Goal: Understand the DGS-Fabeln-1-SE dataset before training anything.
# First, Downloaded the dataset from https://zenodo.org/records/18879038
# and placed ALL CSV files into the `data/` folder.


import sys
sys.path.insert(0, "..")   # so Python can find the src/ folder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import load_tabular, DATA_DIR


# Load the data
df, feature_cols, label_col = load_tabular()
print(f"\nTotal segments  : {len(df)}")
print(f"Total features  : {len(feature_cols)}")
print(f"Label column    : {label_col}")


# Class distribution / is the dataset balanced?
label_counts = df[label_col].value_counts()
print("\nLabel distribution:")
print(label_counts)

plt.figure(figsize=(6, 4))
label_counts.plot(kind="bar", color=["#e74c3c", "#95a5a6", "#2ecc71"])
plt.title("Sentiment Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../outputs/plots/class_distribution.png", dpi=150)
plt.show()


# Feature statistics / look at first few features
print("\nFirst 5 features — statistics:")
print(df[feature_cols[:5]].describe().round(3))


# Check for missing values
missing = df[feature_cols].isna().sum()
n_missing = (missing > 0).sum()
print(f"\nFeatures with missing values: {n_missing} / {len(feature_cols)}")
if n_missing > 0:
    print(missing[missing > 0].head(10))


# Feature correlation heatmap (first 30 features)
plt.figure(figsize=(12, 10))
corr = df[feature_cols[:30]].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.3,
            xticklabels=False, yticklabels=False)
plt.title("Feature Correlation (first 30 features)")
plt.tight_layout()
plt.savefig("../outputs/plots/feature_correlation.png", dpi=150)
plt.show()


# Feature variance by class: which features differ most between sentiments? / This gives a first hint at what the model will use
from scipy import stats

variances = []
for feat in feature_cols[:50]:   # sample first 50
    groups = [df[df[label_col] == c][feat].dropna() for c in df[label_col].unique()]
    try:
        f_stat, p_val = stats.f_oneway(*groups)
        variances.append((feat, f_stat, p_val))
    except Exception:
        pass

var_df = pd.DataFrame(variances, columns=["feature", "F_stat", "p_value"])
var_df = var_df.sort_values("F_stat", ascending=False)
print("\nTop 10 most discriminative features (ANOVA F-statistic):")
print(var_df.head(10).to_string(index=False))

# Box plots for top 3 most discriminative features
top3 = var_df["feature"].values[:3]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, feat in zip(axes, top3):
    df.boxplot(column=feat, by=label_col, ax=ax)
    ax.set_title(feat[:40])
    ax.set_xlabel("Sentiment")
plt.suptitle("Top 3 Discriminative Features by Sentiment")
plt.tight_layout()
plt.savefig("../outputs/plots/top3_features_boxplot.png", dpi=150)
plt.show()

print("\n✓ Exploration complete. Next → run 02_feature_engineering.py")
