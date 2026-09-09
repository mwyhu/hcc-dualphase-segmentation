import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")

df = pd.read_csv("patient_features.csv")
df = df[df["min_HU"] > -2000]


fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(nrows=3, ncols=1)

# Tumour volume distribution by dataset (log scale)
ax1 = fig.add_subplot(gs[0])
df["log_volume"] = np.log10(df["tumour_volume_cm3"] + 1e-5)
sns.kdeplot(
    data=df,
    x="log_volume",
    hue="dataset",
    common_norm=False,
    fill=True,
    alpha=0.3,
    palette="tab10",
    ax=ax1,
)
ax1.set_title(
    "Probability Density (KDE) of Log Tumour Volume Across Datasets"
)
ax1.set_xlabel("Log10(Tumour Volume in cm3)")


# Violin Plot: Mean HU distribution across datasets
ax2 = fig.add_subplot(gs[1])
sns.violinplot(
    data=df,
    x="dataset",
    y="mean_HU",
    hue="dataset",
    palette="Set2",
    inner="quartile",
    ax=ax2,
    legend=False,
)
ax2.set_title("Distribution Shape (Violin Plot) of Mean HU per Dataset")
ax2.tick_params(axis="x", rotation=15)


# Violin Plot: tumour volume across datasets
ax3 = fig.add_subplot(gs[2])
sns.violinplot(
    data=df,
    x="dataset",
    y="tumour_volume_cm3",
    hue="dataset",
    palette="Pastel1",
    inner="quartile",
    ax=ax3,
    legend=False,
)
ax3.set_title("Distribution Shape of Tumour Volume per Dataset")
ax3.set_yscale("log")
ax3.tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("dataset_distribution_analysis.png", dpi=300)