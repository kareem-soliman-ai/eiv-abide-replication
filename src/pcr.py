"""Minimal PCR implementation for the ABIDE-I EIV paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import linregress


@dataclass
class PCRFitResult:
    """Container for a minimal PCR fit."""

    slope: float
    intercept: float
    iterations: int
    converged: bool
    ols_slope: float
    ols_intercept: float
    x_hat: np.ndarray
    weights: np.ndarray
    trace: pd.DataFrame


def weighted_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Fit a weighted least-squares line."""

    weight_sum = float(np.sum(weights))
    x_bar = float(np.sum(weights * x) / weight_sum)
    y_bar = float(np.sum(weights * y) / weight_sum)
    slope = float(
        np.sum(weights * (x - x_bar) * (y - y_bar))
        / np.sum(weights * (x - x_bar) ** 2)
    )
    intercept = float(y_bar - slope * x_bar)
    return slope, intercept


def fit_pcr_regression(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    tolerance: float = 1e-8,
    max_iterations: int = 500,
) -> PCRFitResult:
    """Fit the paper's heteroscedastic PCR model via EM."""

    ols = linregress(x_obs, y_obs)
    slope = float(ols.slope)
    intercept = float(ols.intercept)
    trace_rows: list[dict[str, float | int | bool]] = []

    x_hat = x_obs.copy()
    weights = np.ones_like(x_obs, dtype=float)
    converged = False

    for iteration in range(1, max_iterations + 1):
        denominator = sigma_y**2 + (slope**2) * sigma_x**2
        x_hat = (
            (sigma_y**2 * x_obs)
            + (sigma_x**2 * slope * (y_obs - intercept))
        ) / denominator

        weights = 1.0 / denominator
        next_slope, next_intercept = weighted_least_squares(x_hat, y_obs, weights)
        max_parameter_change = max(
            abs(next_slope - slope),
            abs(next_intercept - intercept),
        )

        trace_rows.append(
            {
                "iteration": iteration,
                "slope": next_slope,
                "intercept": next_intercept,
                "max_parameter_change": max_parameter_change,
                "converged": max_parameter_change < tolerance,
            }
        )

        slope = next_slope
        intercept = next_intercept
        if max_parameter_change < tolerance:
            converged = True
            break

    trace = pd.DataFrame(trace_rows)
    return PCRFitResult(
        slope=slope,
        intercept=intercept,
        iterations=int(len(trace_rows)),
        converged=converged,
        ols_slope=float(ols.slope),
        ols_intercept=float(ols.intercept),
        x_hat=x_hat,
        weights=weights,
        trace=trace,
    )
