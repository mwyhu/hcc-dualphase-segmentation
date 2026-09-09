import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv("patient_features.csv")
df = df[df["min_HU"] >= -2000].copy()

# Source vs Target grouping
source_datasets = ["MSD", "Liver_Lesions", "WAW_TACE"]
df["source_vs_target_group"] = df["dataset"].apply(
    lambda x: "Combined_Source" if x in source_datasets else x
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Counts per dataset by phase
sns.countplot(
    data=df,
    x="dataset",
    hue="phase",
    palette="viridis",
    ax=axes[0],
)
axes[0].set_title("Number of Scans per Dataset (by Phase)")
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend(title="Phase")


# Counts per source vs target group by phase
sns.countplot(
    data=df,
    x="source_vs_target_group",
    hue="phase",
    palette="magma",
    ax=axes[1],
)
axes[1].set_title("Number of Scans: Combined Source vs. Targets (by Phase)")
axes[1].tick_params(axis="x", rotation=15)
axes[1].legend(title="Phase")

plt.tight_layout()
plt.savefig("scanning_counts_by_phase.png", dpi=300)



# HU percentile/spread metrics into long format
hu_profile_cols = ["p5_HU", "median_HU", "p95_HU", "iqr_HU"]
df_melted = df.melt(
    id_vars=["source_vs_target_group"],
    value_vars=hu_profile_cols,
    var_name="HU_Metric",
    value_name="HU_Value",
)

plt.figure(figsize=(14, 7))
sns.boxplot(
    data=df_melted,
    x="source_vs_target_group",
    y="HU_Value",
    hue="HU_Metric",
    palette="Set2",
)
plt.title(
    "Comparison of HU Percentiles and Spread (p5, Median, p95, IQR) Across"
    " Groups"
)
plt.xlabel("Dataset Grouping")
plt.ylabel("Hounsfield Units (HU) / Spread")
plt.xticks(rotation=15)
plt.legend(title="HU Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("hu_percentile_spread_profile.png", dpi=300)

