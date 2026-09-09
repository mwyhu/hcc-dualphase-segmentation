import matplotlib.pyplot as plt
import numpy as np
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
    return row["dataset"]  # Keeps individual target names


df["source_vs_target_group"] = df.apply(label_source_target, axis=1)

# smooth KDE plotting
df["log_volume"] = np.log10(df["tumour_volume_cm3"] + 1e-5)


fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(nrows=2, ncols=1)

# KDE Curve of log volume (Combined Source vs Individual Targets)
ax1 = fig.add_subplot(gs[0])
sns.kdeplot(
    data=df,
    x="log_volume",
    hue="source_vs_target_group",
    common_norm=False,
    fill=True,
    alpha=0.3,
    palette="Set1",
    ax=ax1,
)
ax1.set_title(
    "Volume Density (KDE): Combined Source vs. Individual Target Datasets"
)
ax1.set_xlabel("Log10(Tumour Volume in cm3)")


# Violin Plot of mean HU (Combined Source vs Individual Targets)
ax2 = fig.add_subplot(gs[1])
sns.violinplot(
    data=df,
    x="source_vs_target_group",
    y="mean_HU",
    hue="source_vs_target_group",
    palette="Set2",
    inner="quartile",
    ax=ax2,
    legend=False,
)
ax2.set_title(
    "Tissue Density Shape (Mean HU): Combined Source vs. Individual Target"
    " Datasets"
)
ax2.set_xlabel("Dataset Grouping")
ax2.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("source_vs_target_distributions.png", dpi=300)