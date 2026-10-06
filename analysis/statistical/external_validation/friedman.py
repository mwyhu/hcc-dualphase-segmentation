from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel, friedmanchisquare

DEFAULT_BASE = Path('/Users/michellehu/Desktop/hcc-dualphase-segmentation/analysis/statistical/external_validation')
MODELS = ['SP', 'DP', 'TSLL', 'TSLT', 'AtlasNet']
DATASETS = ['HCC', 'LiTS', 'TCIA', 'WORC']
METRICS = ['dice', 'precision', 'recall', 'detectability']
COMPARISONS = [('DP', 'SP')] + [(model, baseline) for model in ['SP', 'DP'] for baseline in ['TSLL', 'TSLT', 'AtlasNet']]
IOU = 0.2
LABEL = '1'


def case_id(value):
    name = str(value).replace('\\', '/').rsplit('/', 1)[-1]
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return name


def check_frame(frame, path, metrics):
    if frame.empty:
        raise ValueError(f'No cases in {path}')
    if frame.case_id.duplicated().any():
        raise ValueError(f'Duplicate case identifiers in {path}')
    for metric in metrics:
        v = frame[metric].to_numpy(dtype=float)
        if (np.isinf(v) | (np.isfinite(v) & ((v < 0) | (v > 1)))).any():
            raise ValueError(f'Invalid {metric} scores in {path}')


def load_summary(path):
    with path.open() as handle:
        data = json.load(handle)
    rows = []
    for entry in data['metric_per_case']:
        m = entry['metrics'][LABEL]
        tp, fp, fn = (float(m[key]) for key in ['TP', 'FP', 'FN'])
        counts = np.array([tp, fp, fn])
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError(f'Invalid voxel counts in {path}')
        rows.append(dict(case_id=case_id(entry['reference_file']),
                         dice=float(m['Dice']) if m['Dice'] is not None else np.nan,
                         precision=tp / (tp + fp) if tp + fp > 0 else np.nan,
                         recall=tp / (tp + fn) if tp + fn > 0 else np.nan))
    frame = pd.DataFrame(rows)
    check_frame(frame, path, METRICS[:3])
    return frame


def load_detection(path):
    frame = pd.read_csv(path)
    required = {'case', 'iou_threshold', 'detectability'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns in {path}: {required - set(frame.columns)}')
    thresholds = pd.to_numeric(frame.iou_threshold, errors='raise')
    frame = frame[np.isclose(thresholds, IOU, rtol=0, atol=1e-8)].copy()
    if frame['case'].isna().any():
        raise ValueError(f'Missing case identifiers in {path}')
    frame['case_id'] = frame['case'].map(case_id)
    frame['detectability'] = pd.to_numeric(frame.detectability, errors='raise')
    frame = frame[['case_id', 'detectability']]
    check_frame(frame, path, ['detectability'])
    return frame


def load_all(base):
    paths = []
    for model in MODELS:
        detection_code = 'AN' if model == 'AtlasNet' else model
        for dataset in DATASETS:
            paths.append((model, dataset,
                          base / 'summary_files' / f'{model}_{dataset}_summary.json',
                          base / 'detectability_files' / f'Detect_{detection_code}_{dataset}.csv'))
    missing = [str(p) for _, _, summary, detection in paths for p in [summary, detection] if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing input files:\n' + '\n'.join(missing))
    results = []
    for model, dataset, summary_path, detection_path in paths:
        summary = load_summary(summary_path)
        detection = load_detection(detection_path)
        if set(summary.case_id) != set(detection.case_id):
            raise ValueError(f'Summary/detection case mismatch for {model}, {dataset}')
        merged = summary.merge(detection, on='case_id', validate='one_to_one')
        merged['model'] = model
        merged['dataset'] = dataset
        results.append(merged)
    all_data = pd.concat(results, ignore_index=True)
    # All models must cover the same cases, including dataset in the identity.
    expected = None
    for model in MODELS:
        keys = set(map(tuple, all_data.loc[all_data.model == model, ['dataset', 'case_id']].to_numpy()))
        if expected is not None and keys != expected:
            raise ValueError(f'Case coverage differs for {model}; resolve missing cases before comparison.')
        expected = keys
    return all_data


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    maximum = 0.0
    for rank, index in enumerate(order):
        maximum = max(maximum, (len(p) - rank) * p[index])
        adjusted[index] = min(maximum, 1.0)
    return adjusted


def mean(v):
    return float(np.mean(v)) if len(v) else np.nan


def sd(v):
    return float(np.std(v, ddof=1)) if len(v) >= 2 else np.nan


def analyse_scope(all_data, scope, datasets):
    rows, paired_tables = [], []
    for model_a, model_b in COMPARISONS:
        subset = all_data[all_data.dataset.isin(datasets)]
        a = subset[subset.model == model_a]
        b = subset[subset.model == model_b]
        paired = a[['dataset', 'case_id'] + METRICS].merge(
            b[['dataset', 'case_id'] + METRICS], on=['dataset', 'case_id'],
            suffixes=('_a', '_b'), validate='one_to_one')
        paired['scope'] = scope
        paired['model_a'] = model_a
        paired['model_b'] = model_b
        for metric in METRICS:
            paired[f'{metric}_difference_a_minus_b'] = paired[f'{metric}_a'] - paired[f'{metric}_b']
            valid = paired[[f'{metric}_a', f'{metric}_b']].replace([np.inf, -np.inf], np.nan).dropna()
            va = valid[f'{metric}_a'].to_numpy(dtype=float)
            vb = valid[f'{metric}_b'].to_numpy(dtype=float)
            diff = va - vb
            n = len(diff)
            stat, p, lower, upper = np.nan, np.nan, np.nan, np.nan
            if n >= 2:
                margin = t.ppf(0.975, n - 1) * sd(diff) / np.sqrt(n)
                lower, upper = mean(diff) - margin, mean(diff) + margin
                if np.all(diff == 0):
                    stat, p = 0.0, 1.0
                elif np.all(diff == diff[0]):
                    stat, p = np.copysign(np.inf, diff[0]), 0.0
                else:
                    test = ttest_rel(va, vb)
                    stat, p = float(test.statistic), float(test.pvalue)
            rows.append(dict(scope=scope, datasets_included=';'.join(datasets),
                             model_a=model_a, model_b=model_b, metric=metric,
                             iou_threshold=IOU if metric == 'detectability' else np.nan,
                             possible_training_overlap=('TSLT' in [model_a, model_b] and 'LiTS' in datasets),
                             n_paired_cases=n, n_excluded_pairs=len(paired)-n,
                             model_a_mean=mean(va), model_b_mean=mean(vb),
                             model_a_sd=sd(va), model_b_sd=sd(vb),
                             mean_difference_a_minus_b=mean(diff), sd_paired_difference=sd(diff),
                             ci_95_lower=lower, ci_95_upper=upper,
                             t_statistic=stat, paired_t_test_p_value=p))
        paired_tables.append(paired)
    results = pd.DataFrame(rows)
    # One 28-test family per scope: 7 comparisons x 4 metrics.
    results['holm_adjusted_p_value'] = holm(results.paired_t_test_p_value)
    results['significant_after_holm'] = results.holm_adjusted_p_value < 0.05
    return results, pd.concat(paired_tables, ignore_index=True)


def omnibus_scope(all_data, scope, datasets):
    """Friedman repeated-measures tests, using complete cases per metric."""
    subset = all_data[all_data.dataset.isin(datasets)]
    rows = []
    ranks = []
    for metric in METRICS:
        wide = subset.pivot(index=['dataset', 'case_id'], columns='model', values=metric).reindex(columns=MODELS)
        complete = wide.replace([np.inf, -np.inf], np.nan).dropna()
        n = len(complete)
        statistic, p_value = np.nan, np.nan
        if n >= 2:
            values = complete.to_numpy(dtype=float)
            if np.all(values == values[:, :1]):
                # All models tie within every patient: no evidence of a difference.
                statistic, p_value = 0.0, 1.0
            else:
                test = friedmanchisquare(*[complete[model].to_numpy() for model in MODELS])
                statistic, p_value = float(test.statistic), float(test.pvalue)
        rows.append(dict(scope=scope, metric=metric, test='Friedman',
                         datasets_included=';'.join(datasets), n_complete_cases=n,
                         n_excluded_cases=len(wide)-n, n_models=len(MODELS),
                         friedman_statistic=statistic, degrees_of_freedom=len(MODELS)-1,
                         p_value=p_value,
                         kendalls_w=statistic/(n*(len(MODELS)-1)) if n and np.isfinite(statistic) else np.nan,
                         possible_training_overlap='LiTS' in datasets))
        mean_ranks = complete.rank(axis=1, ascending=False, method='average').mean()
        for model in MODELS:
            ranks.append(dict(scope=scope, metric=metric, model=model,
                              n_complete_cases=n, mean_rank=mean_ranks[model]))
    results = pd.DataFrame(rows)
    results['holm_adjusted_p_value'] = holm(results.p_value)
    results['significance'] = results.holm_adjusted_p_value.map(
        lambda p: 'Not testable' if not np.isfinite(p) else ('Significant' if p < 0.05 else 'Not significant'))
    results['significant_after_holm'] = results.holm_adjusted_p_value < 0.05
    return results, pd.DataFrame(ranks)


def descriptive(all_data):
    rows = []
    scopes = [('overall_without_LiTS', ['HCC', 'TCIA', 'WORC']),
              ('overall_all_datasets', DATASETS)] + [(d, [d]) for d in DATASETS]
    for scope, datasets in scopes:
        for model in MODELS:
            frame = all_data[(all_data.model == model) & all_data.dataset.isin(datasets)]
            for metric in METRICS:
                v = frame[metric].dropna()
                rows.append(dict(scope=scope, model=model, metric=metric,
                                 possible_training_overlap=model == 'TSLT' and 'LiTS' in datasets,
                                 n_cases=len(frame), n_valid=len(v), n_missing=len(frame)-len(v),
                                 mean=v.mean(), sd=v.std(ddof=1), median=v.median(),
                                 q1=v.quantile(0.25), q3=v.quantile(0.75)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    data = load_all(args.base_dir)
    output = args.base_dir / 'statistical_results'
    output.mkdir(parents=True, exist_ok=True)
    scopes = [('overall_without_LiTS', ['HCC', 'TCIA', 'WORC']),
              ('overall_all_datasets', DATASETS)] + [(d, [d]) for d in DATASETS]
    results, pairs, omnibus_results, rank_tables = [], [], [], []
    for scope, datasets in scopes:
        omnibus, ranks = omnibus_scope(data, scope, datasets)
        omnibus.to_csv(output / f'omnibus_friedman_{scope}.csv', index=False)
        omnibus_results.append(omnibus)
        rank_tables.append(ranks)
        print(f'\nALL FIVE MODELS TOGETHER: {scope}')
        print(omnibus[['metric', 'n_complete_cases', 'n_excluded_cases', 'friedman_statistic',
                       'holm_adjusted_p_value', 'significance', 'possible_training_overlap']].round(4).to_string(index=False))
        result, paired = analyse_scope(data, scope, datasets)
        result['significance'] = result.holm_adjusted_p_value.map(
            lambda p: 'Not testable' if not np.isfinite(p) else ('Significant' if p < 0.05 else 'Not significant'))
        result.to_csv(output / f'paired_ttest_{scope}.csv', index=False)
        results.append(result)
        pairs.append(paired)
        print(f'\n{scope}: differences are model A minus model B')
        print(result[['model_a', 'model_b', 'metric', 'n_paired_cases',
                      'mean_difference_a_minus_b', 'ci_95_lower', 'ci_95_upper',
                      'holm_adjusted_p_value', 'significance', 'possible_training_overlap']].round(4).to_string(index=False))
    pd.concat(results, ignore_index=True).to_csv(output / 'all_paired_ttest_results.csv', index=False)
    pd.concat(pairs, ignore_index=True).to_csv(output / 'paired_case_level_results.csv', index=False)
    pd.concat(omnibus_results, ignore_index=True).to_csv(output / 'all_omnibus_results.csv', index=False)
    pd.concat(rank_tables, ignore_index=True).to_csv(output / 'model_mean_ranks.csv', index=False)
    data.to_csv(output / 'case_level_metrics.csv', index=False)
    descriptive(data).to_csv(output / 'descriptive_summary.csv', index=False)
    (output / 'analysis_notes.txt').write_text(
        'Omnibus: Friedman tests across all five models, with Holm correction across four metrics per scope.\n'
        'Friedman tests ranks, not means. Mean rank 1 is best. Pairwise t-tests test mean differences.\n'
        'Friedman uses cases valid for all five models per metric; pairwise tests can use different cases.\n'
        'Chi-square approximation: interpret very small complete-case samples cautiously.\n'
        'Planned pairwise tests are reported independently of the omnibus result; they are not rank-based Friedman post-hoc tests.\n'
        'Primary overall comparison: HCC, TCIA and WORC, excluding LiTS consistently for all models.\n'
        'Additional overall comparison: all four datasets. TSLT comparisons including LiTS have possible training overlap.\n'
        'HCC-specific results address the HCC research question directly.\n'
        'Two-sided paired t-tests; differences model A minus model B. Positive differences favour model A.\n'
        'Holm correction: 28 planned tests separately within each scope. No correction across scopes; interpret additional scopes as exploratory.\n'
        '95% confidence intervals are unadjusted. Overall results weight patients equally, not datasets equally.\n'
        'Precision and recall are per-case voxel metrics from summary TP/FP/FN. Undefined denominators produce NaN.\n'
        'Each metric excludes pairs with an undefined value in either model. Descriptive means can use different cases.\n'
        'Detectability reads CSV rows at threshold 0.2; assumes the detection script used IoU >= 0.2.\n'
        'Assumes one independent scan per patient and no unaccounted cross-dataset patient duplicates.\n'
        'Independence from model training must be checked for every evaluated model.\n')
    print(f'\nSaved results to {output}')


if __name__ == '__main__':
    main()
