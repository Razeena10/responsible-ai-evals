"""Statistical helpers shared by the enhanced evaluations.

Every function here is a pure calculation with an explicit formula.  The
intervals are reported as descriptive uncertainty for a synthetic sample; they
are not a substitute for a real-world study design.

Methods implemented:
    - Wilson score interval for a single binomial proportion.
    - Wald (normal-approximation) interval for the difference of two
      independent proportions.
    - Seeded percentile bootstrap interval for the difference of two means
      (used here on 0/1 outcome vectors, i.e. a success-rate difference).

The normality assumptions behind the Wald interval are weakest for small
samples and extreme proportions; the Wilson interval is used for single rates
for that reason.  The bootstrap makes no normal assumption but resamples the
observed data, so it inherits every bias in how those data were generated.
"""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(
    successes: int, total: int, z: float = 1.96
) -> tuple[float, float]:
    """Return the Wilson score interval for ``successes / total``.

    Args:
        successes: Number of successes (0 <= successes <= total).
        total: Number of trials.
        z: Normal quantile (1.96 for a 95% interval).

    Returns:
        ``(lower, upper)`` proportions, clipped to [0, 1].  With no trials the
        interval is ``(0.0, 1.0)``: the data carry no information.

    Raises:
        ValueError: If counts are inconsistent or negative.
    """
    if total < 0 or successes < 0 or successes > total:
        raise ValueError(
            f"invalid counts: successes={successes}, total={total}"
        )
    if total == 0:
        return 0.0, 1.0

    p = successes / total
    denominator = 1.0 + (z * z) / total
    centre = (p + (z * z) / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(p * (1.0 - p) / total + (z * z) / (4.0 * total * total))
        / denominator
    )
    lower = max(0.0, centre - half_width)
    upper = min(1.0, centre + half_width)
    # Clamp floating-point residue (e.g. 6.9e-18) so a zero count reports 0.
    if lower < 1e-12:
        lower = 0.0
    if upper > 1.0 - 1e-12:
        upper = 1.0
    return lower, upper


def wald_difference_interval(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
    z: float = 1.96,
) -> tuple[float, float, float]:
    """Return ``(difference, lower, upper)`` in proportion units.

    ``difference = p_a - p_b``.  The interval is the normal approximation
    ``difference ± z * sqrt(p_a(1-p_a)/n_a + p_b(1-p_b)/n_b)`` and is not
    clipped: a difference interval may legitimately include negative values.

    Raises:
        ValueError: If either sample is empty or counts are inconsistent.
    """
    if total_a <= 0 or total_b <= 0:
        raise ValueError("both samples must be non-empty")
    if not (0 <= successes_a <= total_a) or not (0 <= successes_b <= total_b):
        raise ValueError("invalid counts")

    p_a = successes_a / total_a
    p_b = successes_b / total_b
    difference = p_a - p_b
    standard_error = math.sqrt(
        p_a * (1.0 - p_a) / total_a + p_b * (1.0 - p_b) / total_b
    )
    return difference, difference - z * standard_error, difference + z * standard_error


def bootstrap_difference_interval(
    sample_a: np.ndarray | list[float],
    sample_b: np.ndarray | list[float],
    seed: int,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for ``mean(a) - mean(b)``.

    Args:
        sample_a: Observations for group A (0/1 outcome values here).
        sample_b: Observations for group B.
        seed: Seed for the bootstrap RNG; identical seeds give identical
            intervals.
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level in (0, 1).

    Returns:
        ``(point_estimate, lower, upper)`` in the units of the samples.

    Raises:
        ValueError: If either sample is empty or the arguments are invalid.
    """
    arr_a = np.asarray(sample_a, dtype=float)
    arr_b = np.asarray(sample_b, dtype=float)
    if arr_a.size == 0 or arr_b.size == 0:
        raise ValueError("both samples must be non-empty")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0, 1)")

    rng = np.random.default_rng(seed)
    n_a = arr_a.size
    n_b = arr_b.size
    differences = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resample_a = arr_a[rng.integers(0, n_a, n_a)]
        resample_b = arr_b[rng.integers(0, n_b, n_b)]
        differences[i] = resample_a.mean() - resample_b.mean()

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(differences, [alpha, 1.0 - alpha])
    return float(arr_a.mean() - arr_b.mean()), float(lower), float(upper)


__all__ = [
    "bootstrap_difference_interval",
    "wald_difference_interval",
    "wilson_interval",
]
