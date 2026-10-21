import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = ("dice", "precision", "recall", "detectability")

ALIASES = {
    "WORC": "WORC",
    "WORC_CRLM": "WORC",
    "TCIA": "TCIA",
    "TCIA_CRLM": "TCIA",
    "CAIRO": "CAIRO5",
    "CAIRO5": "CAIRO5",
}

DATASETS = ("WORC", "TCIA", "CAIRO5")


def case_id(value):
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def load_native(base, requested_models, threshold, label):
    """
    Read voxel metrics from nnU-Net summary JSON files and
    lesion detectability from CSV files.

    CSV precision is lesion-level precision and is not used here.
    Voxel precision comes from the summary JSON files.
    """
    summary_dir = base / "summary_files"
    detect_dir = base / "detectability_files"

    models = requested_models
    if not models:
        models = sorted({
            p.name[:-len("_CAIRO5_summary.json")]
            for p in summary_dir.glob("*_CAIRO5_summary.json")
        })

    if not models:
        raise ValueError(
            "No models found; specify --models and check --base-dir"
        )

    scores = {}

    def put(dataset, model, case, metric, value):
        key = (dataset, model, case_id(case), metric)

        if key in scores:
            raise ValueError(f"Duplicate score: {key}")

        if value is not None:
            value = float(value)

            if math.isnan(value):
                value = None
            elif not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid score for {key}: {value}")

        scores[key] = value

    for model in models:
        for dataset in DATASETS:
            path = summary_dir / f"{model}_{dataset}_summary.json"

            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)

            cases = data.get("metric_per_case")
            if not isinstance(cases, list) or not cases:
                raise ValueError(
                    f"{path}: expected nonempty nnU-Net metric_per_case"
                )

            for entry in cases:
                case = (
                    entry.get("reference_file")
                    or entry.get("prediction_file")
                )

                if not case:
                    raise ValueError(
                        f"{path}: missing reference_file/prediction_file"
                    )

                metrics = entry.get("metrics", {})
                selected_label = label

                if selected_label is None:
                    foreground = [k for k in metrics if k != "0"]

                    if len(foreground) != 1:
                        raise ValueError(
                            f"{path}: specify --label; "
                            f"labels={list(metrics)}"
                        )

                    selected_label = foreground[0]

                if selected_label not in metrics:
                    raise ValueError(
                        f"{path}: label {selected_label} not found"
                    )

                values = metrics[selected_label]

                # Recover metrics from TP/FP/FN if needed.
                for target, source, denominator in (
                    ("dice", "Dice", ("TP", "FP", "FN")),
                    ("precision", "Precision", ("TP", "FP")),
                    ("recall", "Recall", ("TP", "FN")),
                ):
                    if source in values:
                        value = values[source]

                    elif all(k in values for k in denominator):
                        tp = float(values["TP"])

                        if target == "dice":
                            den = (
                                2 * tp
                                + float(values["FP"])
                                + float(values["FN"])
                            )
                            value = 2 * tp / den if den else None
                        else:
                            den = tp + float(values[denominator[1]])
                            value = tp / den if den else None

                    else:
                        raise ValueError(
                            f"{path}: cannot calculate {target} for {case}"
                        )

                    put(dataset, model, case, target, value)

            path = detect_dir / f"Detect_{model}_{dataset}.csv"
            count = 0

            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                required = {"case", "iou_threshold", "detectability"}

                if not required <= set(reader.fieldnames or []):
                    raise ValueError(
                        f"{path}: requires {sorted(required)}"
                    )

                for row in reader:
                    if not math.isclose(
                        float(row["iou_threshold"]),
                        threshold,
                        rel_tol=0,
                        abs_tol=1e-8,
                    ):
                        continue

                    raw = row["detectability"].strip()
                    value = (
                        None
                        if raw.lower() in ("", "nan", "na", "none", "null")
                        else raw
                    )

                    put(
                        dataset,
                        model,
                        row["case"],
                        "detectability",
                        value,
                    )
                    count += 1

            if not count:
                raise ValueError(
                    f"{path}: no rows at IoU {threshold}"
                )

    return scores, sorted(models)


def quantile(values, p):
    """Linear interpolation, equivalent to NumPy's default quantile."""
    a = sorted(values)
    position = (len(a) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)

    return a[lo] + (a[hi] - a[lo]) * (position - lo)


def load(paths):
    """Read generic per-case CSVs in wide or long format."""
    scores = {}
    models = set()

    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])

            if not {"dataset", "model", "case_id"} <= columns:
                raise ValueError(
                    f"{path}: requires dataset,model,case_id columns"
                )

            long_format = {"metric", "value"} <= columns

            if not long_format and not set(METRICS) & columns:
                raise ValueError(
                    f"{path}: no supported metric columns"
                )

            for line, row in enumerate(reader, 2):
                dataset = ALIASES.get(
                    row["dataset"].strip().upper()
                )

                if dataset is None:
                    continue

                model = row["model"].strip()
                case = case_id(row["case_id"].strip())

                if not model or not case:
                    raise ValueError(
                        f"{path}:{line}: empty model or case_id"
                    )

                models.add(model)

                if long_format:
                    entries = [
                        (row["metric"].strip().lower(), row["value"])
                    ]
                else:
                    entries = [
                        (metric, row[metric])
                        for metric in METRICS
                        if metric in columns
                    ]

                for metric, raw in entries:
                    if metric not in METRICS:
                        continue

                    key = (dataset, model, case, metric)

                    if key in scores:
                        raise ValueError(
                            f"Duplicate score {key}; "
                            "check overlapping inputs"
                        )

                    raw = (raw or "").strip()

                    if raw.lower() in (
                        "", "nan", "na", "null", "none"
                    ):
                        value = None
                    else:
                        value = float(raw)

                        if (
                            not math.isfinite(value)
                            or not 0 <= value <= 1
                        ):
                            raise ValueError(
                                f"{path}:{line}: "
                                f"{metric} must be in [0,1]"
                            )

                    scores[key] = value

    if not scores:
        raise ValueError("No CRLM scores found")

    missing = set(DATASETS) - {key[0] for key in scores}
    if missing:
        raise ValueError(
            f"Missing CRLM datasets: {sorted(missing)}"
        )

    return scores, sorted(models)


def calculate(scores, models, policy):
    """
    Overall mean:
        Mean of the three dataset means, giving each dataset 1/3 weight.

    Pooled case statistics:
        SD, median and quartiles across all included cases.
        These are separate from the macro-average.

    Case policy:
        common: use the same valid cases across all models per metric.
        available: use each model's available valid cases.
    """
    summary = []
    detail = []
    audit = []

    for metric in METRICS:
        valid = {
            model: {
                (dataset, case)
                for (dataset, m, case, mt), value in scores.items()
                if m == model
                and mt == metric
                and value is not None
            }
            for model in models
        }

        if not any(valid.values()):
            continue

        common = set.intersection(
            *(valid[model] for model in models)
        )

        for model in models:
            selected = (
                common if policy == "common" else valid[model]
            )

            by_dataset = defaultdict(list)

            for dataset, case in sorted(selected):
                value = scores[(dataset, model, case, metric)]
                by_dataset[dataset].append(value)

                audit.append({
                    "dataset": dataset,
                    "model": model,
                    "case_id": case,
                    "metric": metric,
                    "value": value,
                })

            values = [
                value
                for dataset in DATASETS
                for value in by_dataset[dataset]
            ]
            n = len(values)

            if not n:
                raise ValueError(
                    f"No eligible cases for {model}/{metric}; "
                    "check model coverage or use "
                    "--case-policy available"
                )

            missing = [
                dataset
                for dataset in DATASETS
                if not by_dataset[dataset]
            ]

            if missing:
                raise ValueError(
                    f"{model}/{metric}: no eligible cases for "
                    f"{missing}; cannot average all three datasets"
                )

            # Equal dataset weighting.
            dataset_means = {
                dataset: statistics.mean(by_dataset[dataset])
                for dataset in DATASETS
            }
            macro_mean = statistics.mean(dataset_means.values())

            # Descriptive statistics of the pooled cases.
            pooled_q1 = quantile(values, 0.25)
            pooled_q3 = quantile(values, 0.75)

            summary.append({
                "tumour_type": "CRLM",
                "model": model,
                "metric": metric,
                "case_policy": policy,
                "n_datasets": len(DATASETS),
                "n_valid": n,
                "n_excluded_valid": len(valid[model]) - n,
                "macro_mean": macro_mean,
                "pooled_case_std": (
                    statistics.stdev(values) if n > 1 else ""
                ),
                "pooled_case_median": statistics.median(values),
                "pooled_case_q1": pooled_q1,
                "pooled_case_q3": pooled_q3,
                "pooled_case_iqr": pooled_q3 - pooled_q1,
            })

            for dataset in DATASETS:
                a = by_dataset[dataset]
                q1 = quantile(a, 0.25)
                q3 = quantile(a, 0.75)

                detail.append({
                    "dataset": dataset,
                    "model": model,
                    "metric": metric,
                    "case_policy": policy,
                    "n_valid": len(a),
                    "weight": 1 / len(DATASETS),
                    "mean": dataset_means[dataset],
                    "std": (
                        statistics.stdev(a) if len(a) > 1 else ""
                    ),
                    "median": statistics.median(a),
                    "q1": q1,
                    "q3": q3,
                    "iqr": q3 - q1,
                })

    return summary, detail, audit


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate CRLM performance with equal weighting "
            "of WORC, TCIA and CAIRO5."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Optional generic per-case CSVs",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Folder containing summary_files "
            "and detectability_files"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Model filename tokens, e.g. "
            "SP DP TSLL TSLT AtlasNet COALA"
        ),
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.20,
        help="Exact CSV threshold to select",
    )
    parser.add_argument(
        "--label",
        help="Tumour label in summary JSON; auto if one foreground label",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("crlm_results"),
    )
    parser.add_argument(
        "--case-policy",
        choices=["common", "available"],
        default="common",
    )

    args = parser.parse_args()

    try:
        if args.inputs:
            scores, models = load(args.inputs)
        else:
            scores, models = load_native(
                args.base_dir,
                args.models,
                args.iou,
                args.label,
            )

        summary, detail, audit = calculate(
            scores,
            models,
            args.case_policy,
        )

        for rows in (summary, detail, audit):
            for row in rows:
                row["iou_threshold"] = (
                    args.iou
                    if row["metric"] == "detectability"
                    and not args.inputs
                    else ""
                )

    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in (
        ("crlm_summary.csv", summary),
        ("dataset_summary.csv", detail),
        ("included_case_scores.csv", audit),
    ):
        write_csv(args.output_dir / name, rows)

    print(
        "\nOverall macro-mean: "
        "(mean WORC + mean TCIA + mean CAIRO5) / 3\n"
    )

    for row in summary:
        print(
            f"{row['model']:12s} "
            f"{row['metric']:14s} "
            f"n={row['n_valid']:5d} "
            f"macro_mean={row['macro_mean']:.4f}"
        )

    print(
        "\nPooled case median [Q1, Q3] "
        "(separate from the macro-mean):\n"
    )

    for row in summary:
        print(
            f"{row['model']:12s} "
            f"{row['metric']:14s} "
            f"median={row['pooled_case_median']:.4f} "
            f"[{row['pooled_case_q1']:.4f}, "
            f"{row['pooled_case_q3']:.4f}]"
        )

    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()