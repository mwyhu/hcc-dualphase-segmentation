import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


sns.set_theme(style="whitegrid")

df = pd.read_csv("patient_features.csv")
df = df[df["min_HU"] >= -2000].copy()

# Source vs. Target grouping
source_datasets = ["MSD", "Liver_Lesions", "WAW_TACE"]
df["source_vs_target_group"] = df["dataset"].apply(
    lambda x: "Combined_Source" if x in source_datasets else x
)


feature_cols = [
    "tumour_volume_cm3",
    "mean_HU",
    "median_HU",
    "std_HU",
    "min_HU",
    "max_HU",
    "p5_HU",
    "p95_HU",
    "iqr_HU",
]



# Dimensionality Reduction (PCA)
fig1, ax1 = plt.subplots(figsize=(10, 8))

# Standardize features before projection
X = df[feature_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if HAS_UMAP:
  reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
  embedding = reducer.fit_transform(X_scaled)
  proj_name = "UMAP"
else:
  reducer = PCA(n_components=2, random_state=42)
  embedding = reducer.fit_transform(X_scaled)
  proj_name = "PCA"

df["proj_1"] = embedding[:, 0]
df["proj_2"] = embedding[:, 1]

sns.scatterplot(
    data=df,
    x="proj_1",
    y="proj_2",
    hue="source_vs_target_group",
    palette="tab10",
    alpha=0.7,
    s=50,
    ax=ax1,
)
ax1.set_title(
    f"Domain Shift Analysis: {proj_name} Projection of Radiometric Features"
)
ax1.set_xlabel(f"{proj_name} Dimension 1")
ax1.set_ylabel(f"{proj_name} Dimension 2")
ax1.legend(title="Dataset Grouping", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("domain_shift_pca.png", dpi=300)



# Cumulative Distribution Function (CDF)
fig3, ax3 = plt.subplots(figsize=(10, 6))

df["log_volume"] = np.log10(df["tumour_volume_cm3"] + 1e-5)

sns.ecdfplot(
    data=df,
    x="log_volume",
    hue="source_vs_target_group",
    palette="tab10",
    linewidth=2.5,
    ax=ax3,
)
ax3.set_title("Cumulative Distribution Function (CDF) of Tumour Volume")
ax3.set_xlabel("Log10(Tumour Volume in cm3)")
ax3.set_ylabel("Cumulative Probability")

# Move the legend cleanly outside
sns.move_legend(ax3, loc="upper left", bbox_to_anchor=(1.05, 1), title="Dataset Grouping")

plt.tight_layout()
plt.savefig("volume_cdf.png", dpi=300)
