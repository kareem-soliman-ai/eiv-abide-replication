"""8x8 sensitivity analysis for the minimal ABIDE-I replication."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import linregress

from cross_validation import leave_site_out_cross_validation
from pcr import fit_pcr_regression
from utils import (
    attach_uncertainty_columns,
    attenuation_percentage,
    filter_abide_sample,
    load_abide_csv,
    resolve_input_path,
)

MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]


def run_sensitivity_grid(
    frame: pd.DataFrame,
    multipliers: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full 8x8 grid over sigma_x and sigma_y multipliers."""

    multipliers = multipliers or MULTIPLIERS
    ols = linregress(frame["FIQ"], frame["func_mean_fd"])
    ols_slope = float(getattr(ols, "slope"))
    rows: list[dict[str, float]] = []

    for sigma_x_multiplier in multipliers:
        for sigma_y_multiplier in multipliers:
            baseline = attach_uncertainty_columns(
                frame,
                sigma_x_multiplier=sigma_x_multiplier,
                sigma_y_multiplier=sigma_y_multiplier,
            )
            fit = fit_pcr_regression(
                baseline["FIQ"].to_numpy(dtype=float),
                baseline["func_mean_fd"].to_numpy(dtype=float),
                baseline["sigma_x"].to_numpy(dtype=float),
                baseline["sigma_y"].to_numpy(dtype=float),
            )
            _, _, overall_r2 = leave_site_out_cross_validation(
                frame,
                sigma_x_multiplier=sigma_x_multiplier,
                sigma_y_multiplier=sigma_y_multiplier,
            )
            rows.append(
                {
                    "sigma_x_multiplier": sigma_x_multiplier,
                    "sigma_y_multiplier": sigma_y_multiplier,
                    "pcr_slope": fit.slope,
                    "attenuation_pct": attenuation_percentage(ols_slope, float(fit.slope)),
                    "loso_r2": overall_r2,
                }
            )

    long_frame = pd.DataFrame(rows)
    slope_grid = long_frame.pivot(
        index="sigma_x_multiplier",
        columns="sigma_y_multiplier",
        values="pcr_slope",
    )
    attenuation_grid = long_frame.pivot(
        index="sigma_x_multiplier",
        columns="sigma_y_multiplier",
        values="attenuation_pct",
    )
    r2_grid = long_frame.pivot(
        index="sigma_x_multiplier",
        columns="sigma_y_multiplier",
        values="loso_r2",
    )
    return long_frame, slope_grid, attenuation_grid, r2_grid


def main() -> None:
    """Run the sensitivity grid from the command line."""

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
    long_frame, slope_grid, attenuation_grid, r2_grid = run_sensitivity_grid(filtered)

    long_frame.to_csv(output_dir / "sensitivity_grid_long.csv", index=False)
    slope_grid.to_csv(output_dir / "sensitivity_slope_grid.csv")
    attenuation_grid.to_csv(output_dir / "sensitivity_attenuation_grid.csv")
    r2_grid.to_csv(output_dir / "sensitivity_loso_r2_grid.csv")

    print(f"Input: {input_path}")
    print(f"Computed {len(long_frame)} grid points.")


if __name__ == "__main__":
    main()
