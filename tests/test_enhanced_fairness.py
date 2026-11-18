"""Tests for the enhanced fairness evaluation.

Checks reproducibility of the synthetic generator, independent recomputation of
rates and counts, interval helpers, scenario-level and composition
decomposition arithmetic, and report consistency and disclaimers.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from src.enhanced_fairness_evaluator import (
    GROUPS,
    NUM_SCENARIOS,
    RECORDS_PER_GROUP,
    SEEDS,
    generate_seed,
    run_enhanced_fairness_evaluation,
    write_reports,
)
from src.statistics_utils import (
    bootstrap_difference_interval,
    wald_difference_interval,
    wilson_interval,
)


@pytest.fixture(scope="module")
def report() -> dict:
    return run_enhanced_fairness_evaluation()


# ---------------------------------------------------------------------------
# Generation and reproducibility
# ---------------------------------------------------------------------------


def test_generator_is_reproducible():
    first = generate_seed(42)
    second = generate_seed(42)
    for group in GROUPS:
        assert np.array_equal(
            first[group]["scenario"], second[group]["scenario"]
        )
        assert np.array_equal(
            first[group]["success"], second[group]["success"]
        )


def test_different_seeds_differ():
    first = generate_seed(42)
    second = generate_seed(43)
    assert not np.array_equal(
        first["A"]["success"], second["A"]["success"]
    )


def test_record_counts_are_as_documented():
    records = generate_seed(42)
    for group in GROUPS:
        assert records[group]["success"].size == RECORDS_PER_GROUP
        assert set(np.unique(records[group]["scenario"])) <= set(
            range(NUM_SCENARIOS)
        )
        assert set(np.unique(records[group]["success"])) <= {0, 1}


def test_seeds_and_totals(report):
    meta = report["metadata"]
    assert tuple(meta["seeds"]) == SEEDS
    assert meta["records_per_seed"] == RECORDS_PER_GROUP * len(GROUPS)
    assert meta["total_records"] == meta["records_per_seed"] * len(SEEDS)
    assert meta["total_records"] >= 1000
    assert "not a real-world bias audit" in meta["scope_note"]
    assert meta["scope_note"].count("synthetic") >= 1


# ---------------------------------------------------------------------------
# Independent metric calculations
# ---------------------------------------------------------------------------


def test_pooled_rates_match_regenerated_records(report):
    pooled_successes = {group: 0 for group in GROUPS}
    pooled_total = {group: 0 for group in GROUPS}
    for seed in SEEDS:
        records = generate_seed(seed)
        for group in GROUPS:
            pooled_successes[group] += int(records[group]["success"].sum())
            pooled_total[group] += RECORDS_PER_GROUP
    for group in GROUPS:
        block = report["pooled"]["groups"][group]
        assert block["successes"] == pooled_successes[group]
        assert block["total"] == pooled_total[group]
        assert block["success_rate"] == pytest.approx(
            pooled_successes[group] / pooled_total[group]
        )


def test_pooled_difference_matches_manual_calculation(report):
    groups = report["pooled"]["groups"]
    manual = (
        groups["A"]["successes"] / groups["A"]["total"]
        - groups["B"]["successes"] / groups["B"]["total"]
    )
    assert report["pooled"]["difference"][
        "difference_a_minus_b"
    ] == pytest.approx(manual)


def test_per_seed_scenario_counts_sum_to_group_total(report):
    for seed in SEEDS:
        entry = report["per_seed"][str(seed)]
        for group in GROUPS:
            total = sum(
                scenario[group]["total"]
                for scenario in entry["per_scenario"].values()
            )
            assert total == RECORDS_PER_GROUP


def test_scenario_level_rates(report):
    for scenario, entry in report["scenario_level"].items():
        for group in GROUPS:
            block = entry[group]
            assert block["total"] > 0
            assert block["success_rate"] == pytest.approx(
                block["successes"] / block["total"]
            )
            low, high = block["success_rate_95ci"]
            assert 0.0 <= low <= block["success_rate"] <= high <= 1.0
        assert entry["difference"]["difference_a_minus_b"] == pytest.approx(
            entry["A"]["success_rate"] - entry["B"]["success_rate"]
        )


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------


def test_wilson_interval_known_values():
    low, high = wilson_interval(48, 60)
    assert low == pytest.approx(0.682, abs=1e-3)
    assert high == pytest.approx(0.882, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(0, 10)[0] == 0.0


def test_wald_difference_interval_is_symmetric():
    difference, low, high = wald_difference_interval(10, 50, 20, 50)
    assert difference == pytest.approx(-0.2)
    assert difference - low == pytest.approx(high - difference)


def test_bootstrap_is_seeded_and_ordered():
    sample_a = np.array([1] * 30 + [0] * 10)
    sample_b = np.array([1] * 20 + [0] * 20)
    first = bootstrap_difference_interval(
        sample_a, sample_b, seed=7, n_resamples=300
    )
    second = bootstrap_difference_interval(
        sample_a, sample_b, seed=7, n_resamples=300
    )
    assert first == second
    point, low, high = first
    assert low <= high
    assert point == pytest.approx(0.25)


def test_bootstrap_in_report_is_reproducible(report):
    bootstrap = report["pooled"]["difference"]["bootstrap"]
    assert bootstrap["seed"] == 20261117
    low, high = bootstrap["difference_95ci_pp"]
    assert low <= high
    assert low == pytest.approx(
        report["pooled"]["difference"]["difference_95ci_pp"][0], abs=1.5
    )


# ---------------------------------------------------------------------------
# Scenario distribution and decomposition
# ---------------------------------------------------------------------------


def test_composition_decomposition_adds_up(report):
    analysis = report["scenario_distribution_analysis"]
    total = (
        analysis["within_scenario_difference_pp"]
        + analysis["scenario_mix_difference_pp"]
    )
    assert analysis["naive_difference_pp"] == pytest.approx(total, abs=1e-9)
    assert analysis["decomposition_residual_pp"] == pytest.approx(
        0.0, abs=1e-9
    )
    assert analysis["naive_difference_pp"] == pytest.approx(
        report["pooled"]["difference"]["difference_a_minus_b_pp"]
    )


def test_scenario_shares_sum_to_one(report):
    shares = report["scenario_distribution_analysis"][
        "group_scenario_shares"
    ]
    for group in GROUPS:
        assert sum(shares[group].values()) == pytest.approx(1.0)


def test_leave_one_scenario_out(report):
    table = report["leave_one_scenario_out"]
    assert set(table) == {str(s) for s in range(NUM_SCENARIOS)}
    for scenario, entry in table.items():
        for group in GROUPS:
            excluded = report["scenario_level"][scenario][group]["total"]
            assert entry[group]["total"] == (
                RECORDS_PER_GROUP * len(SEEDS) - excluded
            )
        assert entry["difference"]["ci_excludes_zero"] == (
            entry["difference"]["difference_95ci"][0] > 0
            or entry["difference"]["difference_95ci"][1] < 0
        )


# ---------------------------------------------------------------------------
# Report consistency and disclaimers
# ---------------------------------------------------------------------------


def test_reports_written(tmp_path, report):
    json_path, md_path = write_reports(report, str(tmp_path))
    assert os.path.exists(json_path)
    assert os.path.exists(md_path)
    with open(json_path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["metadata"]["total_records"] == (
        report["metadata"]["total_records"]
    )
    with open(md_path, encoding="utf-8") as handle:
        markdown = handle.read()
    assert "real-world bias audit" in markdown
    assert "Synthetic groups" in markdown
    assert "Scenario-level results" in markdown


def test_original_fairness_report_is_untouched(report):
    path = os.path.join("reports", "fairness_report.json")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as handle:
        original = json.load(handle)
    assert original["dataset"]["total_records"] == 120
    assert original["group_results"]["A"]["successful"] == 48
    assert report["comparison_to_original_report"][
        "absolute_difference_pp"
    ] == pytest.approx(15.0)


def test_limitations_are_documented(report):
    limitations = report["limitations"]
    assert len(limitations) >= 5
    joined = " ".join(limitations).lower()
    assert "synthetic" in joined
    assert "parity" in joined
