import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("patient_features.csv")
df = df[df["min_HU"] > -2000]

# Combined Source vs. Target column
source_datasets = ["MSD", "Liver_Lesions", "WAW_TACE"]


def label_source_target(row):
  if row["dataset"] in source_datasets:
    return "Combined_Source"
  else:
    return row["dataset"]


df["source_vs_target_group"] = df.apply(label_source_target, axis=1)

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(nrows=2, ncols=1)


# Mean HU across phases (split by dataset)
ax1 = fig.add_subplot(gs[0])
sns.boxplot(
    data=df,
    x="dataset",
    y="mean_HU",
    hue="phase",
    palette="Set2",
    ax=ax1,
)
ax1.set_title("Mean HU Across Phases for Each Dataset")
ax1.set_xlabel("Dataset")
ax1.set_ylabel("Mean HU")
ax1.tick_params(axis="x", rotation=15)
ax1.legend(title="Phase", bbox_to_anchor=(1.02, 1), loc="upper left")


# Mean HU across phases (combined source vs individual targets)
ax2 = fig.add_subplot(gs[1])
sns.boxplot(
    data=df,
    x="source_vs_target_group",
    y="mean_HU",
    hue="phase",
    palette="muted",
    ax=ax2,
)
ax2.set_title(
    "Mean HU Across Phases: Combined Source vs. Individual Targets"
)
ax2.set_xlabel("Dataset Grouping")
ax2.set_ylabel("Mean HU")
ax2.tick_params(axis="x", rotation=15)
ax2.legend(title="Phase", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig("hu_across_phases_analysis.png", dpi=300)
