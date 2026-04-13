"""Leave-site-out cross-validation for the minimal ABIDE-I replication."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pcr import fit_pcr_regression
from utils import (
    attach_uncertainty_columns,
    filter_abide_sample,
    load_abide_csv,
    manual_r2,
    resolve_input_path,
)


def leave_site_out_cross_validation(
    frame: pd.DataFrame,
    sigma_x_multiplier: float = 1.0,
    sigma_y_multiplier: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Fit on 18 sites and evaluate on the held-out site."""

    prediction_rows: list[dict[str, float | str]] = []
    site_rows: list[dict[str, float | str | int]] = []
    all_true: list[float] = []
    all_pred: list[float] = []

    for site_id in sorted(frame["SITE_ID"].unique()):
        train = frame.loc[frame["SITE_ID"] != site_id].copy()
        test = frame.loc[frame["SITE_ID"] == site_id].copy()
        train = attach_uncertainty_columns(
            train,
            sigma_x_multiplier=sigma_x_multiplier,
            sigma_y_multiplier=sigma_y_multiplier,
        )
        fit = fit_pcr_regression(
            train["FIQ"].to_numpy(dtype=float),
            train["func_mean_fd"].to_numpy(dtype=float),
            train["sigma_x"].to_numpy(dtype=float),
            train["sigma_y"].to_numpy(dtype=float),
        )

        y_true = test["func_mean_fd"].to_numpy(dtype=float)
        y_pred = fit.slope * test["FIQ"].to_numpy(dtype=float) + fit.intercept
        site_r2 = manual_r2(y_true, y_pred)

        for row in test.itertuples(index=False):
            prediction_rows.append(
                {
                    "site_id": site_id,
                    "fiq": float(row.FIQ),
                    "observed_fd": float(row.func_mean_fd),
                    "predicted_fd": float(fit.slope * row.FIQ + fit.intercept),
                }
            )

        site_rows.append(
            {
                "held_out_site": site_id,
                "n_test": int(len(test)),
                "slope_on_train": fit.slope,
                "intercept_on_train": fit.intercept,
                "site_r2": site_r2,
            }
        )

        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

    prediction_frame = pd.DataFrame(prediction_rows)
    site_frame = pd.DataFrame(site_rows)
    overall_r2 = manual_r2(np.asarray(all_true, dtype=float), np.asarray(all_pred, dtype=float))
    return site_frame, prediction_frame, overall_r2


def main() -> None:
    """Run leave-site-out cross-validation from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to the ABIDE-I phenotypic CSV.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "results"),
        help="Directory for CSV outputs.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered = filter_abide_sample(load_abide_csv(input_path))
    per_site, predictions, overall_r2 = leave_site_out_cross_validation(filtered)
    per_site.to_csv(output_dir / "loso_per_site.csv", index=False)
    predictions.to_csv(output_dir / "loso_predictions.csv", index=False)

    print(f"Input: {input_path}")
    print(f"Filtered sample: n={len(filtered)}, sites={filtered['SITE_ID'].nunique()}")
    print(f"Overall LOSO R^2: {overall_r2:.6f}")


if __name__ == "__main__":
    main()
