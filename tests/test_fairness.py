"""Tests for the fairness evaluation module.

Tests cover:
    1. Dataset structure and required columns.
    2. Both groups represented with minimum record counts.
    3. Reproducible data generation.
    4. Correct success/failure counts and rates.
    5. Correct absolute percentage-point difference and success-rate ratio.
    6. Empty group and zero-success-rate handling (no NaN/Infinity).
    7. Threshold configuration.
    8. JSON report validity and no NaN/Infinity.
    9. Markdown and JSON report consistency.
    10. Hand-calculated examples to verify metric formulas.

Responsible AI note: These tests verify the correctness of the metric
computation, not real-world fairness. The groups are synthetic labels.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile

import pytest

from src.fairness_evaluator import (
    FairnessEvaluator,
    GroupMetrics,
    ComparisonMetrics,
    generate_synthetic_data,
    load_dataset,
    run_fairness_evaluation,
    DEFAULT_THRESHOLD_PP,
    RECORDS_PER_GROUP,
    _DEFAULT_DATA_PATH,
    _REQUIRED_COLUMNS,
    _LIMITATIONS,
    _RECOMMENDATIONS,
)
from src.config import DEFAULT_SEED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_records(
    group_a_total: int = 10,
    group_a_success: int = 8,
    group_b_total: int = 10,
    group_b_success: int = 5,
) -> list[dict]:
    """Build a small hand-calculated dataset for formula verification."""
    records = []
    rid = 1
    for _ in range(group_a_success):
        records.append({"record_id": rid, "group": "A", "outcome": 1})
        rid += 1
    for _ in range(group_a_total - group_a_success):
        records.append({"record_id": rid, "group": "A", "outcome": 0})
        rid += 1
    for _ in range(group_b_success):
        records.append({"record_id": rid, "group": "B", "outcome": 1})
        rid += 1
    for _ in range(group_b_total - group_b_success):
        records.append({"record_id": rid, "group": "B", "outcome": 0})
        rid += 1
    return records


# ---------------------------------------------------------------------------
# 1. Dataset structure
# ---------------------------------------------------------------------------


class TestDatasetStructure:
    """Verify the synthetic dataset has the required structure."""

    def test_csv_exists(self):
        assert os.path.exists(_DEFAULT_DATA_PATH)

    def test_has_required_columns(self):
        records = load_dataset()
        for rec in records:
            for col in _REQUIRED_COLUMNS:
                assert col in rec, f"Missing column: {col}"

    def test_record_ids_sequential(self):
        records = load_dataset()
        ids = [r["record_id"] for r in records]
        assert ids == list(range(1, len(records) + 1))

    def test_outcome_is_binary(self):
        records = load_dataset()
        for rec in records:
            assert rec["outcome"] in (0, 1)

    def test_termination_reason_valid(self):
        records = load_dataset()
        valid_reasons = {"resolved", "escalated", "truncated"}
        for rec in records:
            assert rec["termination_reason"] in valid_reasons

    def test_scenario_ids_in_range(self):
        records = load_dataset()
        for rec in records:
            assert 0 <= rec["scenario_id"] < 5


# ---------------------------------------------------------------------------
# 2. Group representation and minimum counts
# ---------------------------------------------------------------------------


class TestGroupRepresentation:
    """Verify both groups are represented with enough records."""

    def test_both_groups_present(self):
        records = load_dataset()
        groups = {r["group"] for r in records}
        assert "A" in groups
        assert "B" in groups

    def test_minimum_records_per_group(self):
        records = load_dataset()
        a_count = sum(1 for r in records if r["group"] == "A")
        b_count = sum(1 for r in records if r["group"] == "B")
        assert a_count >= 50
        assert b_count >= 50

    def test_exact_record_count_per_group(self):
        records = load_dataset()
        a_count = sum(1 for r in records if r["group"] == "A")
        b_count = sum(1 for r in records if r["group"] == "B")
        assert a_count == RECORDS_PER_GROUP
        assert b_count == RECORDS_PER_GROUP

    def test_total_record_count(self):
        records = load_dataset()
        assert len(records) == 2 * RECORDS_PER_GROUP


# ---------------------------------------------------------------------------
# 3. Reproducible data generation
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Verify data generation is deterministic and reproducible."""

    def test_same_seed_same_data(self, tmp_path):
        p1 = str(tmp_path / "data1.csv")
        p2 = str(tmp_path / "data2.csv")
        generate_synthetic_data(p1, seed=42)
        generate_synthetic_data(p2, seed=42)
        with open(p1) as f1, open(p2) as f2:
            assert f1.read() == f2.read()

    def test_different_seed_different_data(self, tmp_path):
        p1 = str(tmp_path / "data1.csv")
        p2 = str(tmp_path / "data2.csv")
        generate_synthetic_data(p1, seed=42)
        generate_synthetic_data(p2, seed=99)
        with open(p1) as f1, open(p2) as f2:
            assert f1.read() != f2.read()

    def test_success_counts_reproducible(self, tmp_path):
        p = str(tmp_path / "data.csv")
        generate_synthetic_data(p, seed=42)
        records = load_dataset(p)
        a_success = sum(1 for r in records if r["group"] == "A" and r["outcome"] == 1)
        b_success = sum(1 for r in records if r["group"] == "B" and r["outcome"] == 1)
        assert a_success == 48
        assert b_success == 39


# ---------------------------------------------------------------------------
# 4. Correct counts and rates
# ---------------------------------------------------------------------------


class TestCountsAndRates:
    """Verify success/failure counts and rates are correct."""

    def test_group_a_counts(self):
        records = load_dataset()
        ev = FairnessEvaluator(records)
        m = ev.compute_group_metrics("A")
        assert m.total == 60
        assert m.successful == 48
        assert m.unsuccessful == 12
        assert m.success_rate == pytest.approx(0.80)
        assert m.failure_rate == pytest.approx(0.20)

    def test_group_b_counts(self):
        records = load_dataset()
        ev = FairnessEvaluator(records)
        m = ev.compute_group_metrics("B")
        assert m.total == 60
        assert m.successful == 39
        assert m.unsuccessful == 21
        assert m.success_rate == pytest.approx(0.65)
        assert m.failure_rate == pytest.approx(0.35)

    def test_success_plus_failure_equals_total(self):
        records = load_dataset()
        ev = FairnessEvaluator(records)
        for group in ("A", "B"):
            m = ev.compute_group_metrics(group)
            assert m.successful + m.unsuccessful == m.total


# ---------------------------------------------------------------------------
# 5. Hand-calculated metric verification
# ---------------------------------------------------------------------------


class TestHandCalculatedMetrics:
    """Verify metric formulas using small hand-calculated examples."""

    def test_known_rates(self):
        """8/10 vs 5/10 → 80% vs 50%, diff=30pp, ratio=0.625."""
        records = _make_records(10, 8, 10, 5)
        ev = FairnessEvaluator(records)
        m_a = ev.compute_group_metrics("A")
        m_b = ev.compute_group_metrics("B")
        comp = ev.compute_comparison()

        assert m_a.success_rate == pytest.approx(0.80)
        assert m_b.success_rate == pytest.approx(0.50)
        assert comp.absolute_difference_pp == pytest.approx(30.0)
        assert comp.success_rate_ratio == pytest.approx(0.625)
        assert comp.higher_success_group == "A"

    def test_equal_rates(self):
        """5/10 vs 5/10 → equal, diff=0pp, ratio=1.0."""
        records = _make_records(10, 5, 10, 5)
        ev = FairnessEvaluator(records)
        comp = ev.compute_comparison()
        assert comp.absolute_difference_pp == pytest.approx(0.0)
        assert comp.success_rate_ratio == pytest.approx(1.0)
        assert comp.higher_success_group == "equal"
        assert comp.exceeds_threshold is False

    def test_one_group_all_success(self):
        """10/10 vs 0/10 → diff=100pp, ratio=0.0."""
        records = _make_records(10, 10, 10, 0)
        ev = FairnessEvaluator(records)
        comp = ev.compute_comparison()
        assert comp.absolute_difference_pp == pytest.approx(100.0)
        assert comp.success_rate_ratio == pytest.approx(0.0)
        assert comp.higher_success_group == "A"

    def test_small_difference_not_significant(self):
        """51/100 vs 49/100 → diff=2pp, does not exceed 5pp threshold."""
        records = _make_records(100, 51, 100, 49)
        ev = FairnessEvaluator(records, threshold_pp=5.0)
        comp = ev.compute_comparison()
        assert comp.absolute_difference_pp == pytest.approx(2.0)
        assert comp.exceeds_threshold is False

    def test_synthetic_dataset_metrics(self):
        """Verify the actual synthetic dataset metrics."""
        records = load_dataset()
        ev = FairnessEvaluator(records)
        comp = ev.compute_comparison()
        assert comp.absolute_difference_pp == pytest.approx(15.0)
        assert comp.success_rate_ratio == pytest.approx(0.65 / 0.80)
        assert comp.higher_success_group == "A"
        assert comp.exceeds_threshold is True


# ---------------------------------------------------------------------------
# 6. Empty group and zero-success-rate handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Verify safe handling of empty groups and zero denominators."""

    def test_empty_group_returns_zeros(self):
        records = _make_records(10, 5, 0, 0)
        ev = FairnessEvaluator(records)
        m_b = ev.compute_group_metrics("B")
        assert m_b.total == 0
        assert m_b.successful == 0
        assert m_b.unsuccessful == 0
        assert m_b.success_rate == 0.0
        assert m_b.failure_rate == 0.0

    def test_both_groups_empty(self):
        records = []
        ev = FairnessEvaluator(records)
        comp = ev.compute_comparison()
        assert comp.absolute_difference_pp == 0.0
        assert comp.success_rate_ratio == 1.0
        assert comp.higher_success_group == "equal"
        assert not math.isnan(comp.absolute_difference_pp)
        assert not math.isnan(comp.success_rate_ratio)

    def test_one_group_zero_success(self):
        """Group A: 0/10, Group B: 5/10 → ratio=0.0, diff=50pp."""
        records = _make_records(10, 0, 10, 5)
        ev = FairnessEvaluator(records)
        comp = ev.compute_comparison()
        assert comp.success_rate_ratio == 0.0
        assert comp.absolute_difference_pp == pytest.approx(50.0)
        assert not math.isnan(comp.success_rate_ratio)

    def test_both_groups_zero_success(self):
        """Both 0% → ratio=1.0 (equal), diff=0.0."""
        records = _make_records(10, 0, 10, 0)
        ev = FairnessEvaluator(records)
        comp = ev.compute_comparison()
        assert comp.success_rate_ratio == 1.0
        assert comp.absolute_difference_pp == 0.0
        assert comp.higher_success_group == "equal"

    def test_no_nan_in_metrics(self):
        """No metric should ever be NaN or Infinity."""
        records = load_dataset()
        ev = FairnessEvaluator(records)
        comp = ev.compute_comparison()
        all_values = [
            comp.absolute_difference_pp,
            comp.success_rate_ratio,
            comp.threshold_pp,
        ]
        for v in all_values:
            assert not math.isnan(v), f"NaN found: {v}"
            assert not math.isinf(v), f"Inf found: {v}"

    def test_empty_group_no_nan(self):
        records = _make_records(0, 0, 0, 0)
        ev = FairnessEvaluator(records)
        comp = ev.compute_comparison()
        assert not math.isnan(comp.absolute_difference_pp)
        assert not math.isnan(comp.success_rate_ratio)


# ---------------------------------------------------------------------------
# 7. Threshold configuration
# ---------------------------------------------------------------------------


class TestThreshold:
    """Verify the configurable threshold works correctly."""

    def test_default_threshold(self):
        records = load_dataset()
        ev = FairnessEvaluator(records)
        assert ev._threshold_pp == DEFAULT_THRESHOLD_PP
        assert ev._threshold_pp == 5.0

    def test_custom_threshold(self):
        records = load_dataset()
        ev = FairnessEvaluator(records, threshold_pp=20.0)
        comp = ev.compute_comparison()
        assert comp.threshold_pp == 20.0
        assert comp.exceeds_threshold is False  # 15pp < 20pp

    def test_threshold_boundary(self):
        """Difference exactly at threshold does not exceed (strict >)."""
        records = _make_records(100, 55, 100, 50)  # 55% vs 50% = 5pp
        ev = FairnessEvaluator(records, threshold_pp=5.0)
        comp = ev.compute_comparison()
        assert comp.absolute_difference_pp == pytest.approx(5.0)
        assert comp.exceeds_threshold is False  # 5.0 > 5.0 is False

    def test_threshold_just_above(self):
        records = _make_records(100, 56, 100, 50)  # 56% vs 50% = 6pp
        ev = FairnessEvaluator(records, threshold_pp=5.0)
        comp = ev.compute_comparison()
        assert comp.exceeds_threshold is True


# ---------------------------------------------------------------------------
# 8. JSON report validity
# ---------------------------------------------------------------------------


class TestJsonReport:
    """Verify the JSON report is valid and has required structure."""

    @pytest.fixture(scope="class")
    @classmethod
    def report_data(cls, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("fairness")
        records = load_dataset()
        ev = FairnessEvaluator(records)
        json_path = str(tmp / "fairness_report.json")
        report = ev.generate_json_report(json_path)
        with open(json_path, encoding="utf-8") as fh:
            json_data = json.load(fh)
        return json_data, json_path

    def test_json_file_exists(self, report_data):
        _, path = report_data
        assert os.path.exists(path)

    def test_json_is_valid_dict(self, report_data):
        data, _ = report_data
        assert isinstance(data, dict)

    def test_has_required_top_level_keys(self, report_data):
        data, _ = report_data
        required = {
            "generated_at", "evaluation_type", "dataset",
            "group_results", "comparison", "metric_definitions",
            "threshold", "interpretation", "limitations", "recommendations",
        }
        assert required <= set(data.keys())

    def test_evaluation_type_is_fairness(self, report_data):
        data, _ = report_data
        assert data["evaluation_type"] == "fairness"

    def test_dataset_has_total_records(self, report_data):
        data, _ = report_data
        assert data["dataset"]["total_records"] == 120

    def test_group_results_have_metrics(self, report_data):
        data, _ = report_data
        for group in ("A", "B"):
            gr = data["group_results"][group]
            assert "total" in gr
            assert "successful" in gr
            assert "unsuccessful" in gr
            assert "success_rate" in gr
            assert "failure_rate" in gr

    def test_comparison_has_metrics(self, report_data):
        data, _ = report_data
        comp = data["comparison"]
        assert "absolute_difference_pp" in comp
        assert "success_rate_ratio" in comp
        assert "higher_success_group" in comp
        assert "threshold_pp" in comp
        assert "exceeds_threshold" in comp

    def test_no_nan_in_json(self, report_data):
        data, _ = report_data
        # Check all numeric values in comparison and group_results.
        comp = data["comparison"]
        for key in ("absolute_difference_pp", "success_rate_ratio", "threshold_pp"):
            v = comp[key]
            assert not math.isnan(v), f"NaN in {key}"
            assert not math.isinf(v), f"Inf in {key}"
        for group in ("A", "B"):
            gr = data["group_results"][group]
            for key in ("success_rate", "failure_rate"):
                v = gr[key]
                assert not math.isnan(v), f"NaN in {group}.{key}"
                assert not math.isinf(v), f"Inf in {group}.{key}"

    def test_json_serializable(self, report_data):
        """The full report dict can be re-serialised to JSON."""
        data, _ = report_data
        json.dumps(data)  # Should not raise

    def test_limitations_present(self, report_data):
        data, _ = report_data
        assert len(data["limitations"]) >= 4
        assert len(data["recommendations"]) >= 4

    def test_metric_definitions_present(self, report_data):
        data, _ = report_data
        defs = data["metric_definitions"]
        assert "success_rate" in defs
        assert "absolute_difference_pp" in defs
        assert "success_rate_ratio" in defs


# ---------------------------------------------------------------------------
# 9. Markdown and JSON consistency
# ---------------------------------------------------------------------------


class TestMarkdownJsonConsistency:
    """Verify the Markdown report matches the JSON report."""

    @pytest.fixture(scope="class")
    @classmethod
    def both_reports(cls, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("fairness")
        records = load_dataset()
        ev = FairnessEvaluator(records)
        json_path = str(tmp / "fairness_report.json")
        md_path = str(tmp / "fairness_report.md")
        report = ev.generate_json_report(json_path)
        ev.generate_markdown_report(md_path)
        with open(json_path, encoding="utf-8") as fh:
            json_data = json.load(fh)
        with open(md_path, encoding="utf-8") as fh:
            md_content = fh.read()
        return json_data, md_content

    def test_markdown_has_title(self, both_reports):
        _, md = both_reports
        assert "# Fairness Evaluation Report" in md

    def test_markdown_has_dataset_section(self, both_reports):
        _, md = both_reports
        assert "## Dataset Description" in md

    def test_markdown_has_group_results(self, both_reports):
        _, md = both_reports
        assert "## Group-Level Outcome Counts and Rates" in md

    def test_markdown_has_comparison(self, both_reports):
        _, md = both_reports
        assert "## Fairness Comparison Metrics" in md

    def test_markdown_has_metric_definitions(self, both_reports):
        _, md = both_reports
        assert "## Metric Definitions and Formulas" in md

    def test_markdown_has_interpretation(self, both_reports):
        _, md = both_reports
        assert "## Interpretation of Observed Results" in md

    def test_markdown_has_limitations(self, both_reports):
        _, md = both_reports
        assert "## Limitations and Assumptions" in md

    def test_markdown_has_recommendations(self, both_reports):
        _, md = both_reports
        assert "## Recommendations for Further Evaluation" in md

    def test_markdown_has_scope_note(self, both_reports):
        _, md = both_reports
        assert "synthetic" in md.lower()
        assert "not a real-world bias audit" in md.lower()

    def test_markdown_contains_json_values(self, both_reports):
        """Key numeric values from JSON appear in Markdown."""
        j, md = both_reports
        comp = j["comparison"]
        # Absolute difference appears.
        assert f"{comp['absolute_difference_pp']:.1f}" in md
        # Success rates appear.
        for group in ("A", "B"):
            rate = j["group_results"][group]["success_rate"]
            # Rate appears as percentage (e.g., "80.0%").
            assert f"{rate:.1%}" in md or f"{rate * 100:.1f}" in md

    def test_markdown_matches_json_total_records(self, both_reports):
        j, md = both_reports
        assert str(j["dataset"]["total_records"]) in md

    def test_markdown_matches_json_threshold(self, both_reports):
        j, md = both_reports
        assert str(j["comparison"]["threshold_pp"]) in md or \
               f"{j['comparison']['threshold_pp']:.1f}" in md


# ---------------------------------------------------------------------------
# 10. Integration: run_fairness_evaluation
# ---------------------------------------------------------------------------


class TestRunFairnessEvaluation:
    """Verify the convenience function works end-to-end."""

    def test_returns_report_dict(self, tmp_path):
        report = run_fairness_evaluation(
            output_dir=str(tmp_path), seed=42
        )
        assert isinstance(report, dict)
        assert report["evaluation_type"] == "fairness"

    def test_creates_json_report(self, tmp_path):
        run_fairness_evaluation(output_dir=str(tmp_path), seed=42)
        assert os.path.exists(os.path.join(str(tmp_path), "fairness_report.json"))

    def test_creates_markdown_report(self, tmp_path):
        run_fairness_evaluation(output_dir=str(tmp_path), seed=42)
        assert os.path.exists(os.path.join(str(tmp_path), "fairness_report.md"))

    def test_json_and_markdown_consistent(self, tmp_path):
        run_fairness_evaluation(output_dir=str(tmp_path), seed=42)
        json_path = os.path.join(str(tmp_path), "fairness_report.json")
        md_path = os.path.join(str(tmp_path), "fairness_report.md")
        with open(json_path) as fh:
            j = json.load(fh)
        with open(md_path) as fh:
            md = fh.read()
        comp = j["comparison"]
        assert f"{comp['absolute_difference_pp']:.1f}" in md
        assert str(j["dataset"]["total_records"]) in md

    def test_custom_threshold_in_report(self, tmp_path):
        report = run_fairness_evaluation(
            output_dir=str(tmp_path), seed=42, threshold_pp=20.0
        )
        assert report["comparison"]["threshold_pp"] == 20.0
        assert report["comparison"]["exceeds_threshold"] is False  # 15pp < 20pp


# ---------------------------------------------------------------------------
# 11. Determinism
# ---------------------------------------------------------------------------


class TestFairnessDeterminism:
    """Verify fairness evaluation is deterministic."""

    def test_same_data_same_metrics(self):
        records = load_dataset()
        ev1 = FairnessEvaluator(records)
        ev2 = FairnessEvaluator(records)
        assert ev1.compute_comparison() == ev2.compute_comparison()

    def test_regenerated_data_same_metrics(self, tmp_path):
        p1 = str(tmp_path / "d1.csv")
        p2 = str(tmp_path / "d2.csv")
        generate_synthetic_data(p1, seed=42)
        generate_synthetic_data(p2, seed=42)
        r1 = load_dataset(p1)
        r2 = load_dataset(p2)
        ev1 = FairnessEvaluator(r1)
        ev2 = FairnessEvaluator(r2)
        assert ev1.compute_comparison() == ev2.compute_comparison()
