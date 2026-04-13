"""Main analysis pipeline for the minimal ABIDE-I replication."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from cross_validation import leave_site_out_cross_validation
from pcr import fit_pcr_regression
from sensitivity_grid import MULTIPLIERS, run_sensitivity_grid
from utils import (
    EXPECTED_HEADLINE,
    EXPECTED_TIER_SLOPES,
    assign_motion_tiers,
    attach_uncertainty_columns,
    bias_factor_from_slopes,
    compute_site_level_summary,
    compute_tier_slopes,
    filter_abide_sample,
    get_project_root,
    load_abide_csv,
    resolve_input_path,
)


def print_heading(title: str) -> None:
    """Print a simple console heading."""

    print(f"\n{title}")
    print("-" * len(title))


def print_convergence_trace(trace: pd.DataFrame) -> None:
    """Print the EM convergence trace."""

    print_heading("EM Convergence Trace")
    for row in trace.itertuples(index=False):
        print(
            f"iter={row.iteration:>3d} "
            f"slope={row.slope:+.12f} "
            f"intercept={row.intercept:+.12f} "
            f"delta={row.max_parameter_change:.3e}"
        )


def verify_results(
    filtered: pd.DataFrame,
    ols,
    pcr_slope: float,
    bias_factor: float,
    overall_loso_r2: float,
    tier_slopes: pd.DataFrame,
) -> None:
    """Assert that the minimal rerun matches the paper-level targets."""

    checks = [
        ("n_subjects", len(filtered), EXPECTED_HEADLINE["n_subjects"], 0),
        ("n_sites", filtered["SITE_ID"].nunique(), EXPECTED_HEADLINE["n_sites"], 0),
        ("ols_slope", float(ols.slope), EXPECTED_HEADLINE["ols_slope"], 1e-5),
        ("pcr_slope", pcr_slope, EXPECTED_HEADLINE["pcr_slope"], 1e-5),
        ("bias_factor", bias_factor, EXPECTED_HEADLINE["bias_factor"], 0.1),
        ("loso_r2", overall_loso_r2, EXPECTED_HEADLINE["loso_r2"], 0.01),
    ]

    for tier_name, expected_slope in EXPECTED_TIER_SLOPES.items():
        observed = float(
            tier_slopes.loc[tier_slopes["tier"] == tier_name, "ols_slope"].iloc[0]
        )
        checks.append((f"{tier_name}_slope", observed, expected_slope, 1e-5))

    for label, observed, expected, tolerance in checks:
        if tolerance == 0:
            assert observed == expected, f"{label}: expected {expected}, observed {observed}"
            continue
        difference = abs(observed - expected)
        assert difference <= tolerance, (
            f"{label}: expected {expected}, observed {observed}, difference {difference}"
        )


def save_headline_results(
    results_dir: Path,
    ols,
    pcr_fit,
    bias_factor: float,
    overall_loso_r2: float,
) -> None:
    """Save the headline table to CSV."""

    headline = pd.DataFrame(
        [
            {"metric": "OLS slope (mm per IQ point)", "value": float(ols.slope)},
            {"metric": "OLS intercept (mm)", "value": float(ols.intercept)},
            {"metric": "PCR slope (mm per IQ point)", "value": float(pcr_fit.slope)},
            {"metric": "PCR intercept (mm)", "value": float(pcr_fit.intercept)},
            {"metric": "Bias factor (OLS overestimate)", "value": bias_factor},
            {"metric": "LOSO R^2", "value": overall_loso_r2},
            {"metric": "EM iterations", "value": int(pcr_fit.iterations)},
        ]
    )
    headline.to_csv(results_dir / "headline_results.csv", index=False)


def save_figure_1(site_summary: pd.DataFrame, figures_dir: Path) -> None:
    """Save Figure 1: site slope versus mean FD bubble chart."""

    plt.figure(figsize=(9, 6))
    bubble_sizes = 40.0 * np.sqrt(site_summary["n_subjects"].to_numpy(dtype=float))
    scatter = plt.scatter(
        site_summary["mean_fd"],
        site_summary["ols_slope"],
        s=bubble_sizes,
        c=site_summary["mean_fd"],
        cmap="viridis",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.6,
    )
    for site_id, mean_fd, slope in zip(
        site_summary["site_id"].astype(str),
        site_summary["mean_fd"].to_numpy(dtype=float),
        site_summary["ols_slope"].to_numpy(dtype=float),
    ):
        plt.annotate(
            site_id,
            (mean_fd, slope),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    plt.colorbar(scatter, label="Site mean FD (mm)")
    plt.xlabel("Site mean FD (mm)")
    plt.ylabel("Within-site OLS slope (mm per IQ point)")
    plt.title("Figure 1. Site-level slope versus mean FD")
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_1_site_slope_vs_mean_fd.png", dpi=200)
    plt.close()


def save_figure_2(
    tier_slopes: pd.DataFrame,
    ols_slope: float,
    pcr_slope: float,
    figures_dir: Path,
) -> None:
    """Save Figure 2: within-tier slopes with pooled reference lines."""

    plt.figure(figsize=(8, 5))
    x_positions = np.arange(len(tier_slopes))
    plt.bar(
        x_positions,
        tier_slopes["ols_slope"],
        color=["#7aa6c2", "#90be6d", "#f9c74f", "#f9844a"],
    )
    plt.axhline(
        ols_slope,
        color="firebrick",
        linestyle="--",
        linewidth=1.5,
        label="Pooled OLS",
    )
    plt.axhline(
        pcr_slope,
        color="navy",
        linestyle="--",
        linewidth=1.5,
        label="PCR",
    )
    plt.xticks(
        x_positions,
        tier_slopes["tier"].astype(str).tolist(),
        rotation=15,
        ha="right",
    )
    plt.ylabel("Within-tier OLS slope (mm per IQ point)")
    plt.title("Figure 2. Within-tier slopes with pooled reference lines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_2_within_tier_slopes.png", dpi=200)
    plt.close()


def save_figure_3(
    frame: pd.DataFrame,
    ols_slope: float,
    ols_intercept: float,
    pcr_slope: float,
    pcr_intercept: float,
    figures_dir: Path,
) -> None:
    """Save Figure 3: full scatter with OLS and PCR lines."""

    plt.figure(figsize=(9, 6))
    colour_map = {
        "Minimal motion": "#577590",
        "Low motion": "#43aa8b",
        "Medium motion": "#f9c74f",
        "High motion": "#f94144",
    }
    for tier_name, tier_frame in frame.groupby("motion_tier", observed=True):
        plt.scatter(
            tier_frame["FIQ"],
            tier_frame["func_mean_fd"],
            s=18,
            alpha=0.65,
            color=colour_map[str(tier_name)],
            label=str(tier_name),
        )

    x_line = np.linspace(frame["FIQ"].min(), frame["FIQ"].max(), 200)
    plt.plot(
        x_line,
        ols_slope * x_line + ols_intercept,
        color="firebrick",
        linewidth=2.0,
        label="OLS",
    )
    plt.plot(
        x_line,
        pcr_slope * x_line + pcr_intercept,
        color="navy",
        linewidth=2.0,
        linestyle="--",
        label="PCR",
    )
    plt.xlabel("Full-scale IQ")
    plt.ylabel("Mean framewise displacement (mm)")
    plt.title("Figure 3. Full scatter with OLS and PCR lines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_3_ols_vs_pcr_overlay.png", dpi=200)
    plt.close()


def save_figure_4(attenuation_grid: pd.DataFrame, figures_dir: Path) -> None:
    """Save Figure 4: sensitivity heatmap."""

    values = attenuation_grid.to_numpy(dtype=float)
    plt.figure(figsize=(8, 6))
    image = plt.imshow(values, cmap="coolwarm", aspect="auto", origin="upper")
    plt.colorbar(image, label="Attenuation relative to PCR (%)")
    plt.xticks(
        np.arange(len(attenuation_grid.columns)),
        [str(label) for label in attenuation_grid.columns],
    )
    plt.yticks(
        np.arange(len(attenuation_grid.index)),
        [str(label) for label in attenuation_grid.index],
    )
    plt.xlabel("sigma_y multiplier")
    plt.ylabel("sigma_x multiplier")
    plt.title("Figure 4. Sensitivity heatmap")
    plt.tight_layout()
    plt.savefig(figures_dir / "figure_4_sensitivity_heatmap.png", dpi=200)
    plt.close()


def main() -> None:
    """Run the full minimal ABIDE-I replication."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to the ABIDE-I phenotypic CSV.")
    parser.add_argument(
        "--results-dir",
        default=str(get_project_root() / "results"),
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(get_project_root() / "figures"),
        help="Directory for PNG outputs.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw = load_abide_csv(input_path)
    filtered = filter_abide_sample(raw)
    filtered = attach_uncertainty_columns(filtered)
    filtered = assign_motion_tiers(filtered)

    ols = linregress(filtered["FIQ"], filtered["func_mean_fd"])
    ols_slope = float(getattr(ols, "slope"))
    ols_intercept = float(getattr(ols, "intercept"))
    pcr_fit = fit_pcr_regression(
        filtered["FIQ"].to_numpy(dtype=float),
        filtered["func_mean_fd"].to_numpy(dtype=float),
        filtered["sigma_x"].to_numpy(dtype=float),
        filtered["sigma_y"].to_numpy(dtype=float),
    )
    bias_factor = bias_factor_from_slopes(ols_slope, float(pcr_fit.slope))

    tier_slopes = compute_tier_slopes(filtered)
    site_summary = compute_site_level_summary(filtered)
    loso_per_site, loso_predictions, overall_loso_r2 = leave_site_out_cross_validation(
        filtered
    )
    (
        sensitivity_long,
        sensitivity_slope_grid,
        sensitivity_attenuation_grid,
        sensitivity_r2_grid,
    ) = run_sensitivity_grid(filtered, multipliers=MULTIPLIERS)

    public_filtered = filtered[
        [
            "SITE_ID",
            "FIQ",
            "func_mean_fd",
            "sigma_x",
            "sigma_y",
            "site_mean_fd",
            "motion_tier",
        ]
    ].rename(columns={"SITE_ID": "site_id", "FIQ": "fiq", "func_mean_fd": "mean_fd"})

    public_filtered.to_csv(results_dir / "filtered_sample.csv", index=False)
    site_summary.to_csv(results_dir / "site_level_summary.csv", index=False)
    tier_slopes.to_csv(results_dir / "tier_slopes.csv", index=False)
    loso_per_site.to_csv(results_dir / "loso_per_site.csv", index=False)
    loso_predictions.to_csv(results_dir / "loso_predictions.csv", index=False)
    pcr_fit.trace.to_csv(results_dir / "em_convergence_trace.csv", index=False)
    sensitivity_long.to_csv(results_dir / "sensitivity_grid_long.csv", index=False)
    sensitivity_slope_grid.to_csv(results_dir / "sensitivity_slope_grid.csv")
    sensitivity_attenuation_grid.to_csv(results_dir / "sensitivity_attenuation_grid.csv")
    sensitivity_r2_grid.to_csv(results_dir / "sensitivity_loso_r2_grid.csv")
    save_headline_results(results_dir, ols, pcr_fit, bias_factor, overall_loso_r2)

    save_figure_1(site_summary, figures_dir)
    save_figure_2(tier_slopes, ols_slope, float(pcr_fit.slope), figures_dir)
    save_figure_3(
        filtered,
        ols_slope,
        ols_intercept,
        float(pcr_fit.slope),
        float(pcr_fit.intercept),
        figures_dir,
    )
    save_figure_4(sensitivity_attenuation_grid, figures_dir)

    print_heading("Sample")
    print(f"Input CSV: {input_path}")
    print(f"Filtered subjects: {len(filtered)}")
    print(f"Sites: {filtered['SITE_ID'].nunique()}")

    print_heading("Headline Results")
    print(f"OLS slope:      {ols_slope:+.8f}")
    print(f"OLS intercept:  {ols_intercept:+.6f}")
    print(f"PCR slope:      {pcr_fit.slope:+.8f}")
    print(f"PCR intercept:  {pcr_fit.intercept:+.6f}")
    print(f"Bias factor:    {bias_factor:.4f}x")
    print(f"LOSO R^2:       {overall_loso_r2:+.6f}")
    print(f"EM iterations:  {pcr_fit.iterations}")

    print_convergence_trace(pcr_fit.trace)

    print_heading("Table 2 Tier Slopes")
    for row in tier_slopes.itertuples(index=False):
        print(
            f"{row.tier:<15} n={row.n_subjects:>3d} "
            f"mean_fd={row.mean_fd:.3f} slope={row.ols_slope:+.6f}"
        )

    print_heading("LOSO Cross-Validation")
    print(f"Overall R^2: {overall_loso_r2:+.6f}")
    for row in loso_per_site.sort_values("held_out_site").itertuples(index=False):
        print(f"{row.held_out_site:<10} n={row.n_test:>3d} R^2={row.site_r2:+.6f}")

    verify_results(
        filtered=filtered,
        ols=ols,
        pcr_slope=float(pcr_fit.slope),
        bias_factor=bias_factor,
        overall_loso_r2=overall_loso_r2,
        tier_slopes=tier_slopes,
    )

    print_heading("Verification")
    print("All paper-level checks passed within tolerance.")
    print(f"Results saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
