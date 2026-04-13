"""Utility helpers for the minimal ABIDE-I replication."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

EXPECTED_HEADLINE = {
    "n_subjects": 935,
    "n_sites": 19,
    "ols_slope": -0.00125,
    "pcr_slope": -0.00027,
    "bias_factor": 4.67,
    "loso_r2": -0.074,
}

EXPECTED_TIER_SLOPES = {
    "Minimal motion": -0.000056,
    "Low motion": -0.000011,
    "Medium motion": -0.000968,
    "High motion": -0.002605,
}

PAPER_TIER_BREAKS = (5, 9, 14, 19)
PAPER_TIER_LABELS = (
    "Minimal motion",
    "Low motion",
    "Medium motion",
    "High motion",
)


def get_project_root() -> Path:
    """Return the root directory of the minimal replication repo."""

    return Path(__file__).resolve().parents[1]


def default_data_path() -> Path:
    """Return the default public CSV location expected by the repo."""

    return get_project_root() / "data" / "abide_phenotypic.csv"


def resolve_input_path(input_path: str | None) -> Path:
    """Resolve the phenotypic CSV path."""

    if input_path:
        return Path(input_path).expanduser().resolve()

    default_path = default_data_path()
    if default_path.exists():
        return default_path

    data_dir = get_project_root() / "data"
    csv_matches = sorted(data_dir.glob("*.csv"))
    if len(csv_matches) == 1:
        return csv_matches[0]

    raise FileNotFoundError(
        "Could not find the ABIDE-I phenotypic CSV. Place it at "
        f"{default_path} or pass --input explicitly."
    )


def load_abide_csv(path: str | Path) -> pd.DataFrame:
    """Load the raw ABIDE-I phenotypic CSV."""

    frame = pd.read_csv(path)
    for column in ("FIQ", "func_mean_fd", "AGE_AT_SCAN"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["SITE_ID"] = frame["SITE_ID"].astype(str)
    return frame


def filter_abide_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the paper's sample filters."""

    qc_fail = frame["qc_rater_1"].fillna("").str.strip().str.lower() == "fail"
    filtered = frame.loc[
        frame["FIQ"].notna()
        & frame["func_mean_fd"].notna()
        & frame["AGE_AT_SCAN"].notna()
        & (frame["FIQ"] > 40)
        & ~qc_fail
    ].copy()
    return filtered


def sigma_x_from_age(age: pd.Series) -> pd.Series:
    """Return age-banded IQ uncertainty from the paper."""

    sigma_x = np.where(age < 13, 4.0, np.where(age < 16, 3.4, 3.0))
    return pd.Series(sigma_x, index=age.index, dtype=float)


def sigma_y_by_site(frame: pd.DataFrame) -> pd.Series:
    """Return within-site FD standard deviation for each subject."""

    site_sd = frame.groupby("SITE_ID")["func_mean_fd"].std(ddof=1)
    return frame["SITE_ID"].map(site_sd).astype(float)


def attach_uncertainty_columns(
    frame: pd.DataFrame,
    sigma_x_multiplier: float = 1.0,
    sigma_y_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Add baseline uncertainty columns, optionally scaled for sensitivity runs."""

    enriched = frame.copy()
    enriched["sigma_x"] = sigma_x_from_age(enriched["AGE_AT_SCAN"]) * sigma_x_multiplier
    enriched["sigma_y"] = sigma_y_by_site(enriched) * sigma_y_multiplier
    enriched["site_mean_fd"] = enriched.groupby("SITE_ID")["func_mean_fd"].transform("mean")
    return enriched


def compute_site_level_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise site-level mean FD, sample size, and within-site OLS slope."""

    rows: list[dict[str, float | int | str]] = []
    for site_id, site_frame in frame.groupby("SITE_ID"):
        ols = linregress(site_frame["FIQ"], site_frame["func_mean_fd"])
        rows.append(
            {
                "site_id": site_id,
                "n_subjects": int(len(site_frame)),
                "mean_fd": float(site_frame["func_mean_fd"].mean()),
                "sigma_y": float(site_frame["sigma_y"].iloc[0]),
                "ols_slope": float(ols.slope),
                "ols_intercept": float(ols.intercept),
            }
        )

    summary = pd.DataFrame(rows).sort_values("mean_fd").reset_index(drop=True)
    return summary


def assign_motion_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign the manuscript's Table 2 motion tiers.

    The paper reports subject counts of 290, 163, 228, and 254 across the four
    tiers. These counts are reproduced by sorting sites by mean FD and keeping
    the contiguous 5/4/5/5 site grouping used in the manuscript tables.
    """

    enriched = frame.copy()
    ordered_sites = (
        enriched.groupby("SITE_ID")["func_mean_fd"].mean().sort_values().index.tolist()
    )
    tier_lookup: dict[str, str] = {}
    start = 0
    for label, stop in zip(PAPER_TIER_LABELS, PAPER_TIER_BREAKS):
        for site_id in ordered_sites[start:stop]:
            tier_lookup[site_id] = label
        start = stop

    enriched["motion_tier"] = enriched["SITE_ID"].map(tier_lookup)
    enriched["motion_tier"] = pd.Categorical(
        enriched["motion_tier"],
        categories=list(PAPER_TIER_LABELS),
        ordered=True,
    )
    return enriched


def compute_tier_slopes(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute the pooled OLS slope within each paper tier."""

    rows: list[dict[str, float | int | str]] = []
    for tier_name in PAPER_TIER_LABELS:
        tier_frame = frame.loc[frame["motion_tier"] == tier_name]
        ols = linregress(tier_frame["FIQ"], tier_frame["func_mean_fd"])
        rows.append(
            {
                "tier": tier_name,
                "n_subjects": int(len(tier_frame)),
                "mean_fd": float(tier_frame["func_mean_fd"].mean()),
                "ols_slope": float(ols.slope),
                "ols_intercept": float(ols.intercept),
            }
        )
    return pd.DataFrame(rows)


def manual_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the standard coefficient of determination without sklearn."""

    residual_sum_of_squares = float(np.sum((y_true - y_pred) ** 2))
    total_sum_of_squares = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - (residual_sum_of_squares / total_sum_of_squares)


def bias_factor_from_slopes(ols_slope: float, pcr_slope: float) -> float:
    """Return the OLS overestimate factor relative to PCR."""

    return abs(ols_slope / pcr_slope)


def attenuation_percentage(ols_slope: float, pcr_slope: float) -> float:
    """Return attenuation relative to the PCR slope."""

    return (bias_factor_from_slopes(ols_slope, pcr_slope) - 1.0) * 100.0
