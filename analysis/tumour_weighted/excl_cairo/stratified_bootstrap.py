from pathlib import Path
import argparse

import numpy as np
import pandas as pd


BASE_DIR = Path(
    "/Users/michellehu/Desktop/"
    "hcc-dualphase-segmentation/analysis/tumour_weighted/excl_cairo"
)

METRICS = ["dice", "precision", "recall", "detectability"]


def load_data(path):
    data = pd.read_csv(path)

    if data["case_id"].isna().any():
        raise ValueError(f"Missing patient identifiers: {path}")

    if data["case_id"].duplicated().any():
        raise ValueError(f"Duplicate patient identifiers: {path}")

    return data


def get_differences(data, metric):
    single = data[f"{metric}_single"].to_numpy(dtype=float)
    dual = data[f"{metric}_dual"].to_numpy(dtype=float)

    valid = np.isfinite(single) & np.isfinite(dual)

    for scores in (single[valid], dual[valid]):
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"{metric}: scores must be within [0, 1].")

    differences = dual[valid] - single[valid]

    if len(differences) < 2:
        raise ValueError(f"{metric}: insufficient valid patient pairs.")

    return differences


def holm_adjust(p_values):
    p_values = np.asarray(p_values)
    order = np.argsort(p_values)
    n_tests = len(p_values)

    adjusted = np.empty(n_tests)
    adjusted[order] = np.minimum(
        1.0,
        np.maximum.accumulate(
            (n_tests - np.arange(n_tests)) * p_values[order]
        ),
    )

    return adjusted


def bootstrap_metric(tcia, worc, n_bootstrap, rng):
    samples = [tcia, worc]

    # Each dataset contributes 50% to the combined difference.
    observed = np.mean([values.mean() for values in samples])

    observed_se = np.sqrt(
        sum(
            values.var(ddof=1) / len(values)
            for values in samples
        ) / 4
    )

    if observed_se == 0:
        raise ValueError("Zero standard error: bootstrap-t is undefined.")

    roots = []

    for start in range(0, n_bootstrap, 1000):
        batch_size = min(1000, n_bootstrap - start)

        bootstrap_means = []
        bootstrap_variances = []

        for values in samples:
            # Resample patients separately within each dataset.
            # Resampling paired differences preserves SP-DP pairing.
            indices = rng.integers(
                0,
                len(values),
                size=(batch_size, len(values)),
            )
            resampled = values[indices]

            bootstrap_means.append(resampled.mean(axis=1))
            bootstrap_variances.append(
                resampled.var(axis=1, ddof=1) / len(values)
            )

        combined_means = (
            bootstrap_means[0] + bootstrap_means[1]
        ) / 2

        bootstrap_se = np.sqrt(
            (
                bootstrap_variances[0]
                + bootstrap_variances[1]
            ) / 4
        )

        if np.any(bootstrap_se == 0):
            raise ValueError(
                "A bootstrap sample has zero standard error. "
                "Inspect sparse or constant differences."
            )

        # Centred bootstrap statistics
        roots.append(
            (combined_means - observed) / bootstrap_se
        )

    roots = np.concatenate(roots)

    lower_quantile, upper_quantile = np.quantile(
        roots, [0.025, 0.975]
    )

    ci_lower = observed - upper_quantile * observed_se
    ci_upper = observed - lower_quantile * observed_se

    observed_statistic = observed / observed_se

    # Approximate two-sided null-centred bootstrap test
    # The plus-one correction prevents a p-value of zero
    p_value = (
        1
        + np.count_nonzero(
            np.abs(roots) >= abs(observed_statistic)
        )
    ) / (n_bootstrap + 1)

    return observed, ci_lower, ci_upper, p_value


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tcia",
        type=Path,
        default=(
            BASE_DIR
            / "normality_checks_TCIA"
            / "paired_case_level_results.csv"
        ),
    )
    parser.add_argument(
        "--worc",
        type=Path,
        default=(
            BASE_DIR
            / "normality_checks_WORC"
            / "paired_case_level_results.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "crlm_bootstrap_results.csv",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261109,
    )

    args = parser.parse_args()

    if args.n_bootstrap < 1000:
        parser.error("Use at least 1,000 bootstrap replicates.")

    tcia_data = load_data(args.tcia)
    worc_data = load_data(args.worc)

    rng = np.random.default_rng(args.seed)
    rows = []

    for metric in METRICS:
        tcia = get_differences(tcia_data, metric)
        worc = get_differences(worc_data, metric)

        mean_difference, ci_lower, ci_upper, p_value = (
            bootstrap_metric(
                tcia,
                worc,
                args.n_bootstrap,
                rng,
            )
        )

        rows.append({
            "metric": metric,
            "n_TCIA": len(tcia),
            "n_WORC": len(worc),
            "mean_difference_TCIA": tcia.mean(),
            "mean_difference_WORC": worc.mean(),
            "combined_mean_difference": mean_difference,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_bootstrap": p_value,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        })

    results = pd.DataFrame(rows)

    # Correct across the four metrics
    results["p_holm"] = holm_adjust(
        results["p_bootstrap"].to_numpy()
    )
    results["significant_holm"] = results["p_holm"] < 0.05

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()