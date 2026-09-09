import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Settings
sns.set_theme(style="whitegrid")

MODEL_ORDER = ['Start1', 'Start2', 'Ceiling', 'TS']
MODEL_PALETTE = dict(
    zip(MODEL_ORDER, sns.color_palette('Set2', len(MODEL_ORDER)))
)


# Summary function
def create_summary(df, group_columns):
    """
    Create a grouped lesion-detection summary.

    - detectability: mean across cases
    - precision: mean across cases
    - TP, FP, FN: summed lesion counts
    - pooled_precision: calculated from summed counts
    - pooled_recall: calculated from summed counts
    - f1_score: calculated from summed counts
    """
    summary = df.groupby(
        group_columns,
        observed=True
    ).agg({
        'detectability': 'mean',
        'precision': 'mean',
        'TP': 'sum',
        'FP': 'sum',
        'FN': 'sum'
    })

    precision_denominator = summary['TP'] + summary['FP']
    recall_denominator = summary['TP'] + summary['FN']
    f1_denominator = (
        2 * summary['TP']
        + summary['FP']
        + summary['FN']
    )

    summary['pooled_precision'] = np.divide(
        summary['TP'],
        precision_denominator,
        out=np.zeros(len(summary), dtype=float),
        where=precision_denominator != 0
    )

    summary['pooled_recall'] = np.divide(
        summary['TP'],
        recall_denominator,
        out=np.zeros(len(summary), dtype=float),
        where=recall_denominator != 0
    )

    summary['f1_score'] = np.divide(
        2 * summary['TP'],
        f1_denominator,
        out=np.zeros(len(summary), dtype=float),
        where=f1_denominator != 0
    )

    return summary.round(3)


# Read CSV files
base_dir = (
    "/Users/michellehu/Desktop/hcc_active_learning/"
    "analysis/detectability"
)

df_start1 = pd.read_csv(
    f"{base_dir}/detectability_start1.csv"
)
df_start1['model'] = 'Start1'

df_start2 = pd.read_csv(
    f"{base_dir}/detectability_start2.csv"
)
df_start2['model'] = 'Start2'

df_ceiling = pd.read_csv(
    f"{base_dir}/detectability_ceiling.csv"
)
df_ceiling['model'] = 'Ceiling'

df_ts = pd.read_csv(
    f"{base_dir}/detectability_TS.csv"
)
df_ts['model'] = 'TS'


# Combine datasets
df_all = pd.concat(
    [
        df_start1,
        df_start2,
        df_ceiling,
        df_ts
    ],
    ignore_index=True
)

df_all['source'] = df_all['case'].str.split('_').str[0]

df_all['model'] = pd.Categorical(
    df_all['model'],
    categories=MODEL_ORDER,
    ordered=True
)

df_all['iou_threshold'] = pd.to_numeric(
    df_all['iou_threshold']
)

print("\n--- Combined Data ---")
print(df_all.head())


# Detectability by model and IoU threshold
plt.figure(figsize=(10, 6))

sns.barplot(
    data=df_all,
    x='iou_threshold',
    y='detectability',
    hue='model',
    hue_order=MODEL_ORDER,
    errorbar=None,
    palette=MODEL_PALETTE
)

plt.title('Detectability by Model Across IoU Thresholds')
plt.xlabel('IoU Threshold')
plt.ylabel('Mean Detectability')
plt.ylim(0, 1)
plt.legend(
    title='Model',
    bbox_to_anchor=(1.02, 1),
    loc='upper left'
)
plt.tight_layout()
plt.show()


# Precision by model and data source
g = sns.catplot(
    data=df_all,
    kind='bar',
    x='iou_threshold',
    y='precision',
    hue='model',
    hue_order=MODEL_ORDER,
    col='source',
    col_wrap=3,
    height=3.5,
    aspect=1.2,
    palette=MODEL_PALETTE,
    errorbar=None
)

g.set_axis_labels(
    'IoU Threshold',
    'Mean Precision'
)
g.set(ylim=(0, 1))
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle(
    'Precision Across Models and Data Sources'
)
plt.show()


# Detectability heatmap
pivot_df = df_all.pivot_table(
    values='detectability',
    index='model',
    columns='iou_threshold',
    aggfunc='mean',
    observed=True
)

plt.figure(figsize=(8, 5))

sns.heatmap(
    pivot_df,
    annot=True,
    cmap='Blues',
    fmt='.3f',
    cbar=True,
    vmin=0,
    vmax=1
)

plt.title('Average Detectability Heatmap')
plt.xlabel('IoU Threshold')
plt.ylabel('Model')
plt.tight_layout()
plt.show()


# Precision trajectory
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df_all,
    x='iou_threshold',
    y='precision',
    hue='model',
    hue_order=MODEL_ORDER,
    style='model',
    style_order=MODEL_ORDER,
    markers=True,
    dashes=False,
    errorbar=None,
    linewidth=2.5,
    markersize=8,
    palette=MODEL_PALETTE
)

plt.title('Precision Across IoU Thresholds')
plt.xlabel('IoU Threshold')
plt.ylabel('Mean Precision')
plt.ylim(0, 1)
plt.legend(
    title='Model',
    bbox_to_anchor=(1.02, 1),
    loc='upper left'
)
plt.tight_layout()
plt.show()


# Detectability by data source
g = sns.catplot(
    data=df_all,
    kind='bar',
    x='iou_threshold',
    y='detectability',
    hue='model',
    hue_order=MODEL_ORDER,
    col='source',
    col_wrap=3,
    height=4,
    aspect=1.2,
    palette=MODEL_PALETTE,
    errorbar='sd'
)

g.set_axis_labels(
    'IoU Threshold',
    'Mean Detectability'
)
g.set(ylim=(0, 1))
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle(
    'Detectability Across Models by Data Source'
)
plt.show()


# Create summary tables
model_threshold_summary = create_summary(
    df_all,
    ['model', 'iou_threshold']
)

print("\n--- Model and IoU Threshold Summary ---")
print(model_threshold_summary)


model_source_thresh_summary = create_summary(
    df_all,
    ['model', 'source', 'iou_threshold']
)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)

print("\n--- Model, Source and IoU Threshold Summary ---")
print(model_source_thresh_summary)


source_model_thresh_summary = create_summary(
    df_all,
    ['source', 'model', 'iou_threshold']
)

print("\n--- Source, Model and IoU Threshold Summary ---")
print(source_model_thresh_summary)


# Convert summary to normal columns
model_threshold_plot = model_threshold_summary.reset_index()


# F1 score
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=model_threshold_plot,
    x='iou_threshold',
    y='f1_score',
    hue='model',
    hue_order=MODEL_ORDER,
    style='model',
    style_order=MODEL_ORDER,
    markers=True,
    dashes=False,
    linewidth=2.5,
    markersize=8,
    palette=MODEL_PALETTE
)

plt.title('Lesion-level F1 Score Across IoU Thresholds')
plt.xlabel('IoU Threshold')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.legend(
    title='Model',
    bbox_to_anchor=(1.02, 1),
    loc='upper left'
)
plt.tight_layout()
plt.show()


# Precision-recall plot
plt.figure(figsize=(8, 7))

sns.scatterplot(
    data=model_threshold_plot,
    x='pooled_recall',
    y='pooled_precision',
    hue='model',
    hue_order=MODEL_ORDER,
    style='model',
    style_order=MODEL_ORDER,
    s=130,
    palette=MODEL_PALETTE
)

# Add the IoU threshold next to each point
for _, row in model_threshold_plot.iterrows():
    plt.annotate(
        f"{row['iou_threshold']:.2f}",
        (
            row['pooled_recall'],
            row['pooled_precision']
        ),
        xytext=(5, 5),
        textcoords='offset points',
        fontsize=8
    )

plt.title('Lesion-level Precision–Recall Trade-off')
plt.xlabel('Pooled Recall')
plt.ylabel('Pooled Precision')
plt.xlim(0, 1.02)
plt.ylim(0, 1.02)
plt.legend(
    title='Model',
    bbox_to_anchor=(1.02, 1),
    loc='upper left'
)
plt.tight_layout()
plt.show()


# F1 score heatmap
f1_pivot = model_threshold_plot.pivot(
    index='model',
    columns='iou_threshold',
    values='f1_score'
)

plt.figure(figsize=(8, 5))

sns.heatmap(
    f1_pivot,
    annot=True,
    cmap='YlGnBu',
    fmt='.3f',
    cbar=True,
    vmin=0,
    vmax=1
)

plt.title('Lesion-level F1 Score Heatmap')
plt.xlabel('IoU Threshold')
plt.ylabel('Model')
plt.tight_layout()
plt.show()


# Results at IoU threshold 0.15
df_015 = df_all[
    np.isclose(df_all['iou_threshold'], 0.15)
].copy()


model_summary_015 = create_summary(
    df_015,
    ['model']
)

print("\n--- Model Summary at IoU 0.15 ---")
print(model_summary_015)


source_model_summary_015 = create_summary(
    df_015,
    ['source', 'model']
)

print("\n--- Source then Model Summary at IoU 0.15 ---")
print(source_model_summary_015)


# F1 score by model at IoU 0.15
model_summary_015_plot = model_summary_015.reset_index()

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=model_summary_015_plot,
    x='model',
    y='f1_score',
    hue='model',
    order=MODEL_ORDER,
    hue_order=MODEL_ORDER,
    palette=MODEL_PALETTE,
    legend=False
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt='%.3f',
        padding=3
    )

plt.title('Lesion-level F1 Score at IoU Threshold 0.15')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()