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
gs = fig.add_gridspec(nrows=2, ncols=2)  # Layout grid


# Dataset vs Dataset Comparison (tumour volume)
ax1 = fig.add_subplot(gs[0, 0])
sns.boxplot(
    data=df, x="dataset", y="tumour_volume_cm3", ax=ax1, palette="Set3"
)
ax1.set_title("Tumour Volume Across All 7 Datasets")
ax1.set_yscale("log")
ax1.tick_params(axis="x", rotation=45)



# Phase distribution across datasets
ax2 = fig.add_subplot(gs[0, 1])
sns.countplot(
    data=df, x="dataset", hue="phase", ax=ax2, palette="viridis"
)
ax2.set_title("Phase Distribution per Dataset")
ax2.tick_params(axis="x", rotation=45)
ax2.legend(title="Phase")


# Combined Source vs Individual Target Datasets (volume)
ax3 = fig.add_subplot(gs[1, :])  # Spans across the entire bottom row
sns.boxplot(
    data=df,
    x="source_vs_target_group",
    y="tumour_volume_cm3",
    hue="phase",
    ax=ax3,
    palette="muted",
)
ax3.set_title(
    "Combined Source (MSD + Liver_Lesions + WAW_TACE) vs. Individual Target"
    " Datasets"
)
ax3.set_yscale("log")
ax3.set_xlabel("Dataset Grouping")
ax3.tick_params(axis="x", rotation=15)
ax3.legend(title="Phase", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig("dataset_phase_source_target_comparison.png", dpi=300)
