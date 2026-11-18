"""Deterministic fairness evaluation module for responsible-ai-evals.

This module generates a small, reproducible synthetic dataset with two groups
(Group A and Group B), computes group-level outcome metrics, and compares
them using clearly defined fairness metrics.

IMPORTANT — Responsible AI scope:
    - The groups are synthetic labels, NOT real demographic or protected groups.
    - This is an illustrative evaluation exercise, not a real-world bias audit.
    - Outcome-rate parity alone does not establish fairness.
    - Passing or failing the illustrative threshold does not imply legal or
      regulatory compliance.
    - The results do not prove real-world demographic fairness or bias.

Usage::

    from src.fairness_evaluator import run_fairness_evaluation
    metrics = run_fairness_evaluation()
    # Writes reports/fairness_report.json and reports/fairness_report.md
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from src.config import DEFAULT_SEED
from src.support_scenarios import SCENARIOS, NUM_SCENARIOS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data"
)
_REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "reports"
)
_DEFAULT_DATA_PATH = os.path.join(_DATA_DIR, "synthetic_fairness_data.csv")

DEFAULT_THRESHOLD_PP: float = 5.0
RECORDS_PER_GROUP: int = 60

# Group A: 48 successes / 60 total = 80.0%
_GROUP_A_SUCCESSES: int = 48
# Group B: 39 successes / 60 total = 65.0%
_GROUP_B_SUCCESSES: int = 39

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "record_id", "group", "scenario_id", "scenario_name",
    "turns", "total_reward", "outcome", "termination_reason",
)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def generate_synthetic_data(
    output_path: str = _DEFAULT_DATA_PATH,
    seed: int = DEFAULT_SEED,
) -> str:
    """Generate the synthetic fairness dataset as a CSV file.

    The dataset is entirely synthetic.  Group A and Group B are arbitrary
    labels — they do not represent real demographic or protected groups.

    Generation method:
        1. 60 records per group (120 total), seeded with *seed*.
        2. Group A has 48 successes (80%), Group B has 39 (65%).
        3. Scenarios cycle through the five existing support scenarios.
        4. Outcomes are shuffled deterministically.
        5. Successful outcomes use optimal resolution (5 turns, +10.5 reward).
        6. Unsuccessful outcomes are either truncated or escalated.

    Args:
        output_path: Destination CSV file path.
        seed: Fixed random seed for reproducibility.

    Returns:
        The path to the generated CSV file.
    """
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    record_id = 1

    group_config = {
        "A": _GROUP_A_SUCCESSES,
        "B": _GROUP_B_SUCCESSES,
    }

    for group in ("A", "B"):
        num_success = group_config[group]
        num_total = RECORDS_PER_GROUP
        num_failure = num_total - num_success

        # Build outcome list and shuffle deterministically.
        outcomes = [1] * num_success + [0] * num_failure
        rng.shuffle(outcomes)

        for i, outcome in enumerate(outcomes):
            scenario_id = i % NUM_SCENARIOS
            scenario = SCENARIOS[scenario_id]

            if outcome == 1:
                turns = len(scenario.required_fields) + 2
                total_reward = 10.5
                termination_reason = "resolved"
            else:
                # Failure: randomly truncated or escalated.
                if rng.random() < 0.5:
                    turns = 15
                    total_reward = round(
                        rng.uniform(-10.0, -3.0), 2
                    )
                    termination_reason = "truncated"
                else:
                    turns = rng.randint(1, 10)
                    total_reward = round(
                        rng.uniform(-2.0, 3.0), 2
                    )
                    termination_reason = "escalated"

            records.append({
                "record_id": record_id,
                "group": group,
                "scenario_id": scenario_id,
                "scenario_name": scenario.name,
                "turns": turns,
                "total_reward": total_reward,
                "outcome": outcome,
                "termination_reason": termination_reason,
            })
            record_id += 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(records)

    return output_path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(path: str = _DEFAULT_DATA_PATH) -> list[dict[str, Any]]:
    """Load the synthetic fairness dataset from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        A list of dicts, one per record.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        records = list(reader)

    if not records:
        return []

    # Validate required columns.
    missing = set(_REQUIRED_COLUMNS) - set(records[0].keys())
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}"
        )

    # Cast numeric fields.
    for rec in records:
        rec["record_id"] = int(rec["record_id"])
        rec["scenario_id"] = int(rec["scenario_id"])
        rec["turns"] = int(rec["turns"])
        rec["total_reward"] = float(rec["total_reward"])
        rec["outcome"] = int(rec["outcome"])

    return records


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupMetrics:
    """Outcome metrics for a single group.

    Attributes:
        group: Group label (e.g. "A" or "B").
        total: Total number of records.
        successful: Number of successful outcomes (outcome == 1).
        unsuccessful: Number of unsuccessful outcomes (outcome == 0).
        success_rate: successful / total (0.0 if total == 0).
        failure_rate: unsuccessful / total (0.0 if total == 0).
    """

    group: str
    total: int
    successful: int
    unsuccessful: int
    success_rate: float
    failure_rate: float


@dataclass(frozen=True)
class ComparisonMetrics:
    """Fairness comparison between two groups.

    Attributes:
        absolute_difference_pp: |rate_A - rate_B| * 100 (percentage points).
        success_rate_ratio: min(rate_A, rate_B) / max(rate_A, rate_B)
            when max > 0; 1.0 if both rates are 0.0; 0.0 if exactly one is 0.0.
        higher_success_group: Label of the group with the higher success rate.
            "equal" if both rates are identical.
        threshold_pp: Configurable illustrative threshold in percentage points.
        exceeds_threshold: True if absolute_difference_pp > threshold_pp.
    """

    absolute_difference_pp: float
    success_rate_ratio: float
    higher_success_group: str
    threshold_pp: float
    exceeds_threshold: bool


# ---------------------------------------------------------------------------
# FairnessEvaluator
# ---------------------------------------------------------------------------


class FairnessEvaluator:
    """Deterministic fairness evaluation harness.

    Computes group-level outcome metrics and fairness comparison metrics
    from a list of records.  Handles empty groups and zero denominators
    safely — never produces NaN or Infinity.

    Args:
        records: List of record dicts with at least "group" and "outcome" keys.
        threshold_pp: Illustrative threshold in percentage points for flagging
            a meaningful difference.  Defaults to 5.0.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        threshold_pp: float = DEFAULT_THRESHOLD_PP,
    ) -> None:
        self._records = records
        self._threshold_pp = threshold_pp

    def compute_group_metrics(self, group: str) -> GroupMetrics:
        """Compute outcome metrics for a single group.

        Args:
            group: Group label to filter by.

        Returns:
            GroupMetrics with counts and rates.  If the group has zero
            records, all rates are 0.0.
        """
        group_records = [
            r for r in self._records if r.get("group") == group
        ]
        total = len(group_records)
        successful = sum(1 for r in group_records if r.get("outcome") == 1)
        unsuccessful = total - successful
        success_rate = successful / total if total > 0 else 0.0
        failure_rate = unsuccessful / total if total > 0 else 0.0

        return GroupMetrics(
            group=group,
            total=total,
            successful=successful,
            unsuccessful=unsuccessful,
            success_rate=success_rate,
            failure_rate=failure_rate,
        )

    def compute_comparison(
        self,
        group_a: str = "A",
        group_b: str = "B",
    ) -> ComparisonMetrics:
        """Compute fairness comparison metrics between two groups.

        Args:
            group_a: Label for the first group.
            group_b: Label for the second group.

        Returns:
            ComparisonMetrics with absolute difference, ratio, and threshold.
        """
        m_a = self.compute_group_metrics(group_a)
        m_b = self.compute_group_metrics(group_b)

        rate_a = m_a.success_rate
        rate_b = m_b.success_rate

        # Absolute difference in percentage points.  Rounded to 10 decimal
        # places to remove binary floating-point noise (e.g. 5.000000000000004)
        # so a difference exactly at the threshold is not flagged as exceeding
        # it.  The rounding is far below any meaningful precision for rates.
        diff_pp = round(abs(rate_a - rate_b) * 100.0, 10)

        # Success-rate ratio (always finite, never NaN).
        max_rate = max(rate_a, rate_b)
        min_rate = min(rate_a, rate_b)
        if max_rate > 0.0:
            ratio = min_rate / max_rate
        elif min_rate == 0.0:
            # Both rates are 0.0 — they are equal.
            ratio = 1.0
        else:
            # Should not happen, but guard anyway.
            ratio = 0.0

        # Higher success group.
        if rate_a > rate_b:
            higher = group_a
        elif rate_b > rate_a:
            higher = group_b
        else:
            higher = "equal"

        return ComparisonMetrics(
            absolute_difference_pp=diff_pp,
            success_rate_ratio=ratio,
            higher_success_group=higher,
            threshold_pp=self._threshold_pp,
            exceeds_threshold=diff_pp > self._threshold_pp,
        )

    def evaluate(
        self,
        group_a: str = "A",
        group_b: str = "B",
    ) -> dict:
        """Run the full evaluation and return a results dict.

        Args:
            group_a: Label for the first group.
            group_b: Label for the second group.

        Returns:
            A dict with group_metrics and comparison metrics.
        """
        m_a = self.compute_group_metrics(group_a)
        m_b = self.compute_group_metrics(group_b)
        comp = self.compute_comparison(group_a, group_b)

        return {
            "group_metrics": {
                group_a: asdict(m_a),
                group_b: asdict(m_b),
            },
            "comparison": asdict(comp),
        }

    # --- Report generation ------------------------------------------------

    def generate_json_report(
        self,
        path: str,
        data_path: str = _DEFAULT_DATA_PATH,
        seed: int = DEFAULT_SEED,
    ) -> dict:
        """Generate a JSON fairness report and return the report dict.

        Args:
            path: Destination file path for the JSON report.
            data_path: Path to the dataset CSV (for metadata).
            seed: Seed used for data generation (for metadata).
        """
        results = self.evaluate()
        m_a = results["group_metrics"]["A"]
        m_b = results["group_metrics"]["B"]
        comp = results["comparison"]

        # Interpretation is shared with the Markdown report so both outputs
        # describe the metrics with identical wording.
        interpretation = self._build_interpretation(comp, m_a, m_b)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_type": "fairness",
            "dataset": {
                "path": os.path.relpath(
                    data_path, os.path.dirname(os.path.dirname(__file__))
                ),
                "total_records": len(self._records),
                "groups": ["A", "B"],
                "seed": seed,
                "generation_method": (
                    "Deterministic synthetic generation: "
                    f"{RECORDS_PER_GROUP} records per group, outcomes "
                    "shuffled with a fixed seed. Group A has "
                    f"{_GROUP_A_SUCCESSES}/{RECORDS_PER_GROUP} successes "
                    f"({_GROUP_A_SUCCESSES / RECORDS_PER_GROUP:.0%}), "
                    f"Group B has {_GROUP_B_SUCCESSES}/{RECORDS_PER_GROUP} "
                    f"({_GROUP_B_SUCCESSES / RECORDS_PER_GROUP:.0%}). "
                    "Scenarios cycle through the five existing support "
                    "scenarios. Groups are synthetic labels, not real "
                    "demographic groups."
                ),
            },
            "group_results": {
                "A": m_a,
                "B": m_b,
            },
            "comparison": comp,
            "metric_definitions": {
                "success_rate": "successful / total (0.0 if total == 0)",
                "failure_rate": "unsuccessful / total (0.0 if total == 0)",
                "absolute_difference_pp": (
                    "|rate_A - rate_B| * 100 (percentage points)"
                ),
                "success_rate_ratio": (
                    "min(rate_A, rate_B) / max(rate_A, rate_B) when "
                    "max > 0; 1.0 if both rates are 0.0; 0.0 if exactly "
                    "one is 0.0"
                ),
                "higher_success_group": (
                    "Label of the group with the higher success rate; "
                    "'equal' if identical"
                ),
                "exceeds_threshold": (
                    "absolute_difference_pp > threshold_pp"
                ),
            },
            "threshold": {
                "value_pp": self._threshold_pp,
                "meaning": (
                    "Illustrative threshold for flagging a potentially "
                    "meaningful difference. Not a legal or regulatory "
                    "standard."
                ),
            },
            "interpretation": interpretation,
            "limitations": _LIMITATIONS,
            "recommendations": _RECOMMENDATIONS,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)

        return report

    def generate_markdown_report(
        self,
        path: str,
        data_path: str = _DEFAULT_DATA_PATH,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Generate a human-readable Markdown fairness report.

        Args:
            path: Destination .md file path.
            data_path: Path to the dataset CSV (for metadata).
            seed: Seed used for data generation (for metadata).
        """
        results = self.evaluate()
        m_a = results["group_metrics"]["A"]
        m_b = results["group_metrics"]["B"]
        comp = results["comparison"]

        lines: list[str] = []
        a = lines.append

        a("# Fairness Evaluation Report — responsible-ai-evals")
        a("")
        a(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        a("")
        a("> **Scope note:** The groups in this evaluation are synthetic "
          "labels (Group A, Group B). They do not represent real demographic "
          "or protected populations. This is an illustrative evaluation "
          "exercise, not a real-world bias audit. Outcome-rate parity alone "
          "does not establish fairness.")
        a("")
        a("---")
        a("")

        # Dataset description
        a("## Dataset Description")
        a("")
        a(f"| Attribute | Value |")
        a(f"|---|---|")
        a(f"| Total records | {len(self._records)} |")
        a(f"| Groups | A, B |")
        a(f"| Records per group | {RECORDS_PER_GROUP} |")
        a(f"| Seed | {seed} |")
        a(f"| Outcome definition | 1 = successful resolution, 0 = unsuccessful |")
        a("")

        # Data generation method
        a("## Synthetic Data-Generation Method")
        a("")
        a(
            "The dataset was generated deterministically using a fixed random "
            f"seed ({seed}). Group A has {_GROUP_A_SUCCESSES} successes out "
            f"of {RECORDS_PER_GROUP} records "
            f"({_GROUP_A_SUCCESSES / RECORDS_PER_GROUP:.1%}), and Group B "
            f"has {_GROUP_B_SUCCESSES} successes out of "
            f"{RECORDS_PER_GROUP} "
            f"({_GROUP_B_SUCCESSES / RECORDS_PER_GROUP:.1%}). "
            "Scenarios cycle through the five existing customer-support "
            "scenarios. Outcomes are shuffled deterministically. Successful "
            "outcomes use optimal resolution (5 turns, +10.5 reward). "
            "Unsuccessful outcomes are either truncated or escalated. "
            "The groups are synthetic labels — they do not represent real "
            "demographic or protected groups."
        )
        a("")

        # Group-level results
        a("## Group-Level Outcome Counts and Rates")
        a("")
        a("| Group | Total | Successful | Unsuccessful | Success Rate | Failure Rate |")
        a("|---|---|---|---|---|---|")
        a(f"| A | {m_a['total']} | {m_a['successful']} | {m_a['unsuccessful']} "
          f"| {m_a['success_rate']:.1%} | {m_a['failure_rate']:.1%} |")
        a(f"| B | {m_b['total']} | {m_b['successful']} | {m_b['unsuccessful']} "
          f"| {m_b['success_rate']:.1%} | {m_b['failure_rate']:.1%} |")
        a("")

        # Fairness comparison metrics
        a("## Fairness Comparison Metrics")
        a("")
        a("| Metric | Value |")
        a("|---|---|")
        a(f"| Absolute difference | {comp['absolute_difference_pp']:.1f} pp |")
        a(f"| Success-rate ratio | {comp['success_rate_ratio']:.4f} |")
        a(f"| Higher success group | {comp['higher_success_group']} |")
        a(f"| Threshold | {comp['threshold_pp']:.1f} pp |")
        a(f"| Exceeds threshold | {comp['exceeds_threshold']} |")
        a("")

        # Metric definitions
        a("## Metric Definitions and Formulas")
        a("")
        a("| Metric | Formula |")
        a("|---|---|")
        a("| Success rate | successful / total (0.0 if total == 0) |")
        a("| Failure rate | unsuccessful / total (0.0 if total == 0) |")
        a("| Absolute difference (pp) | \\|rate_A - rate_B\\| × 100 |")
        a("| Success-rate ratio | min(rate_A, rate_B) / max(rate_A, rate_B) "
          "when max > 0; 1.0 if both 0.0; 0.0 if exactly one is 0.0 |")
        a("| Higher success group | Group with the higher rate; 'equal' if identical |")
        a("| Exceeds threshold | absolute_difference_pp > threshold_pp |")
        a("")

        # Threshold
        a("## Configurable Threshold")
        a("")
        a(
            f"The illustrative threshold is set to {comp['threshold_pp']:.1f} "
            "percentage points. This is a configurable value for flagging a "
            "potentially meaningful difference. It is not a legal or "
            "regulatory standard."
        )
        a("")

        # Interpretation
        a("## Interpretation of Observed Results")
        a("")
        a(self._build_interpretation(comp, m_a, m_b))
        a("")

        # Limitations
        a("## Limitations and Assumptions")
        a("")
        for i, lim in enumerate(_LIMITATIONS, 1):
            a(f"{i}. {lim}")
        a("")

        # Recommendations
        a("## Recommendations for Further Evaluation")
        a("")
        for i, rec in enumerate(_RECOMMENDATIONS, 1):
            a(f"{i}. {rec}")
        a("")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def _build_interpretation(
        self, comp: dict, m_a: dict, m_b: dict
    ) -> str:
        """Build the interpretation string from comparison metrics."""
        if comp["higher_success_group"] == "equal":
            return (
                "Both groups have identical success rates. "
                "No disparity was observed in this synthetic dataset."
            )
        exceeds = "exceeds" if comp["exceeds_threshold"] else "does not exceed"
        if comp["higher_success_group"] == "A":
            higher_rate = m_a["success_rate"]
            lower_rate = m_b["success_rate"]
        else:
            higher_rate = m_b["success_rate"]
            lower_rate = m_a["success_rate"]
        return (
            f"Group {comp['higher_success_group']} has a higher observed "
            f"success rate ({higher_rate:.1%} vs {lower_rate:.1%}). "
            f"The absolute difference is {comp['absolute_difference_pp']:.1f} "
            f"percentage points, which {exceeds} the illustrative threshold "
            f"of {comp['threshold_pp']:.1f} percentage points. "
            f"This difference is illustrative and does not by itself "
            f"establish unfairness or bias. Outcome-rate parity alone "
            f"does not establish fairness. A small synthetic difference "
            f"should not be used to label a group as disadvantaged without "
            f"further investigation."
        )


# ---------------------------------------------------------------------------
# Static report content
# ---------------------------------------------------------------------------

_LIMITATIONS: list[str] = [
    "The groups are synthetic labels (Group A, Group B). They do not "
    "represent real demographic, protected, or marginalised populations.",

    "The dataset is small (120 records) and generated deterministically. "
    "It does not capture the complexity of real customer-support "
    "interactions or real-world variation.",

    "Outcome-rate parity alone does not establish fairness. Fairness "
    "requires considering context, causation, confounders, and "
    "domain-specific definitions of equity.",

    "This evaluation does not test for intersectional bias, calibration "
    "across subgroups, or disparate impact under legal standards.",

    "Passing or failing the illustrative threshold does not imply legal "
    "or regulatory compliance.",

    f"The synthetic success rates "
    f"({_GROUP_A_SUCCESSES / RECORDS_PER_GROUP:.0%} and "
    f"{_GROUP_B_SUCCESSES / RECORDS_PER_GROUP:.0%}) were chosen to create a "
    "meaningful difference for illustration. They were not derived from "
    "real data or real customer outcomes.",
]

_RECOMMENDATIONS: list[str] = [
    "Expand the evaluation to include more groups, subgroups, and "
    "intersectional analysis once real (de-identified) data is available.",

    "Investigate the root cause of any observed disparity: is it due to "
    "scenario distribution, agent behavior, environment design, or "
    "confounding factors?",

    "Consider additional fairness metrics beyond outcome parity: "
    "calibration, equalised odds, demographic parity, and procedural "
    "fairness.",

    "Validate findings against real (de-identified) data before drawing "
    "any conclusions about real-world bias or fairness.",

    "Engage domain experts, affected communities, and legal advisors "
    "before interpreting synthetic results as evidence of real-world "
    "bias or fairness.",
]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_fairness_evaluation(
    output_dir: str | None = None,
    seed: int = DEFAULT_SEED,
    threshold_pp: float = DEFAULT_THRESHOLD_PP,
) -> dict:
    """Run the full fairness evaluation and generate reports.

    Generates the synthetic dataset (if it does not exist), loads it,
    computes metrics, and writes JSON and Markdown reports.

    Args:
        output_dir: Directory for report files. Defaults to reports/.
        seed: Random seed for data generation.
        threshold_pp: Illustrative threshold in percentage points.

    Returns:
        The full report dict.
    """
    data_path = _DEFAULT_DATA_PATH

    # Generate data if it doesn't exist.
    if not os.path.exists(data_path):
        generate_synthetic_data(data_path, seed=seed)

    records = load_dataset(data_path)
    evaluator = FairnessEvaluator(records, threshold_pp=threshold_pp)

    if output_dir is None:
        output_dir = _REPORTS_DIR

    json_path = os.path.join(output_dir, "fairness_report.json")
    md_path = os.path.join(output_dir, "fairness_report.md")

    report = evaluator.generate_json_report(
        json_path, data_path=data_path, seed=seed
    )
    evaluator.generate_markdown_report(
        md_path, data_path=data_path, seed=seed
    )

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate fairness evaluation reports from the CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate fairness evaluation reports."
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD_PP,
        help=f"Illustrative threshold in pp (default: {DEFAULT_THRESHOLD_PP})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: reports/)",
    )
    args = parser.parse_args()

    # ASCII-only console output: Windows terminals may use cp1252, which
    # cannot encode box-drawing characters or em dashes.
    print("Fairness Evaluation - responsible-ai-evals")
    print(f"  Seed       : {args.seed}")
    print(f"  Threshold  : {args.threshold} pp")
    print(f"  Output dir : {args.output_dir or _REPORTS_DIR}")
    print()

    report = run_fairness_evaluation(
        output_dir=args.output_dir,
        seed=args.seed,
        threshold_pp=args.threshold,
    )

    m_a = report["group_results"]["A"]
    m_b = report["group_results"]["B"]
    comp = report["comparison"]

    print("  ── Results ─────────────────────────────────────")
    print(f"  Total records    : {m_a['total'] + m_b['total']}")
    print(f"  Group A success  : {m_a['successful']}/{m_a['total']} = {m_a['success_rate']:.1%}")
    print(f"  Group B success  : {m_b['successful']}/{m_b['total']} = {m_b['success_rate']:.1%}")
    print(f"  Abs difference    : {comp['absolute_difference_pp']:.1f} pp")
    print(f"  Success ratio    : {comp['success_rate_ratio']:.4f}")
    print(f"  Higher group     : {comp['higher_success_group']}")
    print(f"  Exceeds threshold : {comp['exceeds_threshold']}")
    print("  ────────────────────────────────────────────────")
    print(f"  Reports saved to {args.output_dir or _REPORTS_DIR}/")


if __name__ == "__main__":
    main()
