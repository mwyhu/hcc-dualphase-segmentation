from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------
DEFAULT_BASE = Path(
    '/Users/michellehu/Desktop/hcc-dualphase-segmentation/'
    'analysis/statistical/external_validation'
)
MODELS = ['SP', 'DP', 'TSLL', 'TSLT', 'AtlasNet', 'COALA']
DATASETS = ['HCC', 'LiTS', 'TCIA', 'WORC', 'CAIRO5']
METRICS = ['dice', 'precision', 'recall', 'detectability']
COMPARISONS = [
    (MODELS[row], MODELS[column])
    for row in range(len(MODELS))
    for column in range(row)
]
SCOPES = [(dataset, [dataset]) for dataset in DATASETS]
IOU_THRESHOLD = 0.20
TUMOUR_LABEL = '1'
ALPHA = 0.05


# -------------------------------------------------------------------
# Load and validate
# -------------------------------------------------------------------
def case_id(value):
    name = str(value).replace('\\', '/').rsplit('/', 1)[-1]
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return name


def validate_frame(frame, path, metrics):
    if frame.empty:
        raise ValueError(f'No cases found in {path}')
    if frame['case_id'].duplicated().any():
        raise ValueError(f'Duplicate case identifiers in {path}')
    for metric in metrics:
        values = frame[metric].to_numpy(dtype=float)
        invalid = np.isinf(values) | (
            np.isfinite(values) & ((values < 0) | (values > 1))
        )
        if invalid.any():
            raise ValueError(f'Invalid {metric} scores in {path}')


def load_summary(path):
    with path.open() as handle:
        data = json.load(handle)
    rows = []
    for entry in data['metric_per_case']:
        metrics = entry['metrics'][TUMOUR_LABEL]
        tp, fp, fn = [float(metrics[key]) for key in ['TP', 'FP', 'FN']]
        counts = np.array([tp, fp, fn])
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError(f'Invalid voxel counts in {path}')
        rows.append({
            'case_id': case_id(entry['reference_file']),
            'dice': float(metrics['Dice']) if metrics['Dice'] is not None else np.nan,
            'precision': tp / (tp + fp) if tp + fp > 0 else np.nan,
            'recall': tp / (tp + fn) if tp + fn > 0 else np.nan,
        })
    frame = pd.DataFrame(rows)
    validate_frame(frame, path, METRICS[:3])
    return frame


def load_detection(path):
    frame = pd.read_csv(path)
    required = {'case', 'iou_threshold', 'detectability'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns in {path}: {required - set(frame.columns)}')
    thresholds = pd.to_numeric(frame['iou_threshold'], errors='raise')
    frame = frame[np.isclose(thresholds, IOU_THRESHOLD, rtol=0, atol=1e-8)].copy()
    if frame['case'].isna().any():
        raise ValueError(f'Missing case identifiers in {path}')
    frame['case_id'] = frame['case'].map(case_id)
    frame['detectability'] = pd.to_numeric(frame['detectability'], errors='raise')
    frame = frame[['case_id', 'detectability']]
    validate_frame(frame, path, ['detectability'])
    return frame


def load_all(base):
    paths = []
    for model in MODELS:
        detection_code = 'AN' if model == 'AtlasNet' else model
        for dataset in DATASETS:
            paths.append((
                model, dataset,
                base / 'summary_files' / f'{model}_{dataset}_summary.json',
                base / 'detectability_files' / f'Detect_{detection_code}_{dataset}.csv',
            ))
    missing = [
        str(path)
        for _, _, summary, detection in paths
        for path in [summary, detection]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError('Missing input files:\n' + '\n'.join(missing))
    frames = []
    for model, dataset, summary_path, detection_path in paths:
        summary = load_summary(summary_path)
        detection = load_detection(detection_path)
        if set(summary['case_id']) != set(detection['case_id']):
            raise ValueError(f'Summary/detection case mismatch: {model}, {dataset}')
        frame = summary.merge(detection, on='case_id', validate='one_to_one')
        frame['model'] = model
        frame['dataset'] = dataset
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    expected = None
    for model in MODELS:
        keys = set(map(tuple, data.loc[
            data['model'] == model, ['dataset', 'case_id']
        ].to_numpy()))
        if expected is not None and keys != expected:
            raise ValueError(f'Case coverage differs for {model}. Resolve before comparing.')
        expected = keys
    return data


# -------------------------------------------------------------------
# Statistical helpers
# -------------------------------------------------------------------

def holm(p_values):
    """Holm correction; retain the full configured family size."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    order = valid[np.argsort(p_values[valid])]
    running_maximum = 0.0
    for rank, index in enumerate(order):
        running_maximum = max(
            running_maximum, (len(p_values) - rank) * p_values[index]
        )
        adjusted[index] = min(running_maximum, 1.0)
    return adjusted


def mean(values):
    return float(np.mean(values)) if len(values) else np.nan


def sd(values):
    return float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan


def median(values):
    return float(np.median(values)) if len(values) else np.nan


# -------------------------------------------------------------------
# Friedman and Post Hoc analysis
# -------------------------------------------------------------------

def analyse_scope(data, scope, datasets):
    subset = data[data['dataset'].isin(datasets)]
    omnibus_rows, rank_rows, pairwise_rows, case_tables = [], [], [], []

    for metric in METRICS:
        wide = subset.pivot(
            index=['dataset', 'case_id'], columns='model', values=metric
        ).reindex(columns=MODELS)
        complete = wide.replace([np.inf, -np.inf], np.nan).dropna(subset=MODELS)
        values = complete.to_numpy(dtype=float)
        n = len(complete)

        statistic, p_value = np.nan, np.nan
        if n >= 2:
            if np.all(values == values[:, :1]):
                statistic, p_value = 0.0, 1.0
            else:
                test = friedmanchisquare(*[
                    complete[model].to_numpy() for model in MODELS
                ])
                statistic, p_value = float(test.statistic), float(test.pvalue)

        omnibus_rows.append({
            'scope': scope,
            'datasets_included': ';'.join(datasets),
            'metric': metric,
            'n_complete_cases': n,
            'n_excluded_cases': len(wide) - n,
            'n_models': len(MODELS),
            'friedman_statistic': statistic,
            'degrees_of_freedom': len(MODELS) - 1,
            'friedman_p_value': p_value,
            'kendalls_w': (
                statistic / (n * (len(MODELS) - 1))
                if n and np.isfinite(statistic) else np.nan
            ),
        })

        mean_ranks = complete.rank(axis=1, ascending=False, method='average').mean()
        for model in MODELS:
            rank_rows.append({
                'scope': scope, 'metric': metric, 'model': model,
                'n_complete_cases': n, 'mean_rank': mean_ranks[model],
            })

        for model_a, model_b in COMPARISONS:
            va = complete[model_a].to_numpy(dtype=float)
            vb = complete[model_b].to_numpy(dtype=float)
            differences = va - vb
            test_differences = np.round(differences, decimals=12)
            pair_statistic, pair_p = np.nan, np.nan
            if n >= 2:
                if np.all(test_differences == 0):
                    pair_statistic, pair_p = 0.0, 1.0
                else:
                    test = wilcoxon(
                        test_differences, alternative='two-sided',
                        zero_method='wilcox', method='auto',
                    )
                    pair_statistic, pair_p = float(test.statistic), float(test.pvalue)
            pairwise_rows.append({
                'scope': scope,
                'datasets_included': ';'.join(datasets),
                'metric': metric,
                'model_a': model_a,
                'model_b': model_b,
                'iou_threshold': IOU_THRESHOLD if metric == 'detectability' else np.nan,
                'n_paired_cases': n,
                'n_excluded_cases': len(wide) - n,
                'n_nonzero_differences': int(np.count_nonzero(test_differences)),
                'model_a_mean': mean(va), 'model_b_mean': mean(vb),
                'model_a_sd': sd(va), 'model_b_sd': sd(vb),
                'model_a_median': median(va), 'model_b_median': median(vb),
                'mean_difference_a_minus_b': mean(differences),
                'median_paired_difference_a_minus_b': median(differences),
                'wilcoxon_statistic': pair_statistic,
                'wilcoxon_p_value': pair_p,
            })
            cases = complete[[model_a, model_b]].reset_index().rename(
                columns={model_a: 'score_a', model_b: 'score_b'}
            )
            cases['scope'] = scope
            cases['metric'] = metric
            cases['model_a'] = model_a
            cases['model_b'] = model_b
            cases['difference_a_minus_b'] = differences
            case_tables.append(cases)

    omnibus = pd.DataFrame(omnibus_rows)
    omnibus['holm_adjusted_p_value'] = holm(omnibus['friedman_p_value'])
    omnibus['significant_after_holm'] = omnibus['holm_adjusted_p_value'] < ALPHA
    omnibus['significance'] = omnibus['holm_adjusted_p_value'].map(
        lambda p: 'Not testable' if not np.isfinite(p)
        else ('Significant' if p < ALPHA else 'Not significant')
    )

    pairwise = pd.DataFrame(pairwise_rows)
    pairwise['holm_adjusted_p_value'] = holm(pairwise['wilcoxon_p_value'])
    lookup = omnibus.set_index('metric')['holm_adjusted_p_value']
    pairwise['friedman_holm_adjusted_p_value'] = pairwise['metric'].map(lookup)
    pairwise['friedman_significant_after_holm'] = (
        pairwise['friedman_holm_adjusted_p_value'] < ALPHA
    )
    pairwise['pairwise_significant_after_holm'] = pairwise['holm_adjusted_p_value'] < ALPHA
    pairwise['significant_after_holm'] = (
        pairwise['friedman_significant_after_holm']
        & pairwise['pairwise_significant_after_holm']
    )

    def label(row):
        if not np.isfinite(row['wilcoxon_p_value']) or not np.isfinite(
            row['friedman_holm_adjusted_p_value']
        ):
            return 'Not testable'
        if not row['friedman_significant_after_holm']:
            return 'Not interpreted: Friedman not significant'
        return 'Significant' if row['significant_after_holm'] else 'Not significant'

    pairwise['significance'] = pairwise.apply(label, axis=1)
    return omnibus, pairwise, pd.DataFrame(rank_rows), pd.concat(case_tables, ignore_index=True)


# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

def descriptive_summary(data):
    rows = []
    for scope, datasets in SCOPES:
        for model in MODELS:
            frame = data[(data['model'] == model) & data['dataset'].isin(datasets)]
            for metric in METRICS:
                values = frame[metric].dropna()
                rows.append({
                    'scope': scope, 'model': model, 'metric': metric,
                    'n_cases': len(frame), 'n_valid': len(values),
                    'n_missing': len(frame) - len(values),
                    'mean': values.mean(), 'sd': values.std(ddof=1),
                    'median': values.median(),
                    'q1': values.quantile(0.25), 'q3': values.quantile(0.75),
                })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    data = load_all(args.base_dir)
    output = args.base_dir / 'friedman_posthoc_results'
    output.mkdir(parents=True, exist_ok=True)
    omnibus_tables, pairwise_tables, rank_tables, case_tables = [], [], [], []

    for scope, datasets in SCOPES:
        omnibus, pairwise, ranks, cases = analyse_scope(data, scope, datasets)
        omnibus.to_csv(output / f'omnibus_friedman_{scope}.csv', index=False)
        pairwise.to_csv(output / f'posthoc_wilcoxon_{scope}.csv', index=False)
        omnibus_tables.append(omnibus)
        pairwise_tables.append(pairwise)
        rank_tables.append(ranks)
        case_tables.append(cases)
        print(f'\nFRIEDMAN: {scope}')
        print(omnibus[[
            'metric', 'n_complete_cases', 'n_excluded_cases',
            'friedman_statistic', 'kendalls_w',
            'holm_adjusted_p_value', 'significance',
        ]].to_string(index=False, float_format=lambda v: f'{v:.6g}'))
        print(f'\nPOST HOC WILCOXON: {scope}')
        print('Differences are model A minus model B.')
        print(pairwise[[
            'model_a', 'model_b', 'metric', 'n_paired_cases',
            'mean_difference_a_minus_b', 'median_paired_difference_a_minus_b',
            'holm_adjusted_p_value', 'significance',
        ]].to_string(index=False, float_format=lambda v: f'{v:.6g}'))

    for tables, filename in [
        (omnibus_tables, 'all_omnibus_results.csv'),
        (pairwise_tables, 'all_posthoc_wilcoxon_results.csv'),
        (rank_tables, 'model_mean_ranks.csv'),
        (case_tables, 'paired_case_level_results.csv'),
    ]:
        pd.concat(tables, ignore_index=True).to_csv(output / filename, index=False)
    data.to_csv(output / 'case_level_metrics.csv', index=False)
    descriptive_summary(data).to_csv(output / 'descriptive_summary.csv', index=False)

    print(f'\nSaved results to {output}')


if __name__ == '__main__':
    main()
