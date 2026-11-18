"""Tests for the safety evaluation harness.

Tests cover:
    1. Evaluator structure and test case generation.
    2. Metrics computation correctness (including denominators).
    3. Specific test case results (prohibited, valid, invalid, malformed, bypass).
    4. JSON and Markdown report generation and consistency.
    5. Edge cases (empty evaluation, metric correctness with known inputs).
    6. Determinism and reproducibility.

Both positive and negative cases are included. If the evaluation discovers a
genuine safety weakness, the test preserves and reports it rather than
weakening assertions.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from src import config
from src.safety_evaluator import (
    SafetyEvaluator,
    SafetyTestCase,
    run_safety_evaluation,
    CATEGORY_UNSAFE_ACTION,
    CATEGORY_REPEATED_FAILURE,
    CATEGORY_UNSUPPORTED_INPUT,
    CATEGORY_RULE_BYPASS,
    _StuckAgent,
    _make_observation,
    _run_episode_tracked,
    _WEAKNESSES,
    _LIMITATIONS,
    _MITIGATIONS,
)
from src.guardrails import (
    GuardrailPolicy,
    RULE_ACTION_ALLOWED,
    RULE_INVALID_ACTION_ID,
    RULE_PROHIBITED_ACTION,
    RULE_MALFORMED_OBSERVATION,
)
from src.config import (
    NUM_ACTIONS,
    ESCALATE,
    ASK_BILLING_INFO,
    ASK_ACCOUNT_INFO,
    ASK_TECHNICAL_INFO,
    DIAGNOSE,
    PROPOSE_SOLUTION,
    REWARD_REPEATED_ACTION,
    REWARD_TURN_PENALTY,
)
from src.support_scenarios import SCENARIOS, NUM_SCENARIOS
from src.support_env import CustomerSupportEnv
from src.baseline_agent import BaselineSupportAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def evaluation_results():
    """Run a full safety evaluation once for all tests in this class."""
    evaluator = SafetyEvaluator(seed=42)
    evaluator.run_all()
    return evaluator


# ---------------------------------------------------------------------------
# 1. Evaluator structure
# ---------------------------------------------------------------------------


class TestEvaluatorStructure:
    """Verify the evaluator produces the expected structure."""

    def test_produces_test_cases(self, evaluation_results):
        assert len(evaluation_results.test_cases) > 0

    def test_all_cases_are_safety_test_case(self, evaluation_results):
        for tc in evaluation_results.test_cases:
            assert isinstance(tc, SafetyTestCase)

    def test_all_categories_present(self, evaluation_results):
        categories = {tc.category for tc in evaluation_results.test_cases}
        assert CATEGORY_UNSAFE_ACTION in categories
        assert CATEGORY_REPEATED_FAILURE in categories
        assert CATEGORY_UNSUPPORTED_INPUT in categories
        assert CATEGORY_RULE_BYPASS in categories

    def test_test_ids_unique(self, evaluation_results):
        ids = [tc.test_id for tc in evaluation_results.test_cases]
        assert len(ids) == len(set(ids))

    def test_test_case_has_all_fields(self, evaluation_results):
        tc = evaluation_results.test_cases[0]
        required = {
            "test_id", "category", "description", "proposed_action",
            "observation_description", "guardrail_allowed",
            "guardrail_final_action", "guardrail_policy_rule",
            "expected_behavior", "actual_behavior", "passed",
            "risk_identified", "explanation",
        }
        assert required <= set(tc.__dict__.keys())

    def test_expected_case_count(self, evaluation_results):
        """Verify the minimum expected number of test cases per category."""
        by_cat = {}
        for tc in evaluation_results.test_cases:
            by_cat.setdefault(tc.category, 0)
            by_cat[tc.category] += 1
        assert by_cat[CATEGORY_UNSAFE_ACTION] >= 15
        assert by_cat[CATEGORY_REPEATED_FAILURE] >= 5
        assert by_cat[CATEGORY_UNSUPPORTED_INPUT] >= 8
        assert by_cat[CATEGORY_RULE_BYPASS] >= 4


# ---------------------------------------------------------------------------
# 2. Metrics computation
# ---------------------------------------------------------------------------


class TestSafetyMetrics:
    """Verify metrics are computed correctly with proper denominators."""

    @pytest.fixture(scope="class")
    def metrics(self, evaluation_results):
        return evaluation_results.compute_metrics()

    def test_total_equals_sum(self, metrics):
        assert metrics["total_test_cases"] > 0
        assert (
            metrics["passed_cases"] + metrics["failed_cases"]
            == metrics["total_test_cases"]
        )

    def test_block_rate_has_denominator(self, metrics):
        br = metrics["block_rate_prohibited_actions"]
        assert br["total_prohibited"] > 0
        assert br["blocked"] <= br["total_prohibited"]
        if br["total_prohibited"] > 0:
            expected = br["blocked"] / br["total_prohibited"]
            assert metrics["block_rate_prohibited_actions"]["rate"] == pytest.approx(expected)

    def test_acceptance_rate_has_denominator(self, metrics):
        sr = metrics["safe_action_acceptance_rate"]
        assert sr["total_valid"] > 0
        assert sr["allowed"] <= sr["total_valid"]
        if sr["total_valid"] > 0:
            expected = sr["allowed"] / sr["total_valid"]
            assert metrics["safe_action_acceptance_rate"]["rate"] == pytest.approx(expected)

    def test_unsupported_inputs_has_denominator(self, metrics):
        ui = metrics["unsupported_inputs_handled_safely"]
        assert ui["total"] > 0
        assert ui["handled"] <= ui["total"]

    def test_repeated_action_risks(self, metrics):
        rr = metrics["repeated_action_risks_identified"]
        assert rr["total_repeat_cases"] > 0
        assert rr["risks"] <= rr["total_repeat_cases"]

    def test_invalid_action_block_rate(self, metrics):
        ir = metrics["invalid_action_block_rate"]
        assert ir["total_invalid"] > 0
        assert ir["blocked"] <= ir["total_invalid"]

    def test_rule_bypass_block_rate(self, metrics):
        rb = metrics["rule_bypass_block_rate"]
        assert rb["total"] > 0
        assert rb["blocked"] <= rb["total"]

    def test_metrics_are_consistent_with_cases(self, evaluation_results, metrics):
        """Manually verify the block rate from test cases."""
        prohibited = [
            tc for tc in evaluation_results.test_cases
            if tc.category == CATEGORY_UNSAFE_ACTION
            and tc.guardrail_policy_rule == RULE_PROHIBITED_ACTION
        ]
        blocked = sum(1 for tc in prohibited if not tc.guardrail_allowed)
        assert metrics["block_rate_prohibited_actions"]["blocked"] == blocked
        assert metrics["block_rate_prohibited_actions"]["total_prohibited"] == len(prohibited)


# ---------------------------------------------------------------------------
# 3. Specific test case results
# ---------------------------------------------------------------------------


class TestSafetyTestCases:
    """Verify specific test case outcomes."""

    def test_prohibited_actions_blocked(self, evaluation_results):
        """All prohibited action test cases must be blocked."""
        prohibited = [
            tc for tc in evaluation_results.test_cases
            if tc.category == CATEGORY_UNSAFE_ACTION
            and tc.guardrail_policy_rule == RULE_PROHIBITED_ACTION
        ]
        assert len(prohibited) > 0
        for tc in prohibited:
            assert tc.guardrail_allowed is False, (
                f"{tc.test_id}: prohibited action was allowed"
            )
            assert tc.guardrail_final_action == ESCALATE
            assert tc.passed

    def test_valid_actions_allowed(self, evaluation_results):
        """All valid available action test cases must be allowed."""
        valid = [
            tc for tc in evaluation_results.test_cases
            if tc.category == CATEGORY_UNSAFE_ACTION
            and tc.guardrail_policy_rule == RULE_ACTION_ALLOWED
        ]
        assert len(valid) > 0
        for tc in valid:
            assert tc.guardrail_allowed is True, (
                f"{tc.test_id}: valid action was blocked"
            )
            assert tc.passed

    def test_invalid_actions_blocked(self, evaluation_results):
        """All invalid action test cases must be blocked."""
        invalid = [
            tc for tc in evaluation_results.test_cases
            if tc.category == CATEGORY_UNSAFE_ACTION
            and tc.guardrail_policy_rule == RULE_INVALID_ACTION_ID
        ]
        assert len(invalid) > 0
        for tc in invalid:
            assert tc.guardrail_allowed is False
            assert tc.guardrail_final_action == ESCALATE
            assert tc.passed

    def test_malformed_observations_blocked(self, evaluation_results):
        """All malformed observation test cases must be blocked safely."""
        malformed = [
            tc for tc in evaluation_results.test_cases
            if tc.category == CATEGORY_UNSUPPORTED_INPUT
        ]
        assert len(malformed) > 0
        for tc in malformed:
            assert tc.guardrail_allowed is False
            assert tc.guardrail_policy_rule == RULE_MALFORMED_OBSERVATION
            assert tc.passed

    def test_rule_bypass_blocked(self, evaluation_results):
        """All rule-bypass test cases must be blocked."""
        bypass = [
            tc for tc in evaluation_results.test_cases
            if tc.category == CATEGORY_RULE_BYPASS
        ]
        assert len(bypass) > 0
        for tc in bypass:
            assert tc.guardrail_allowed is False, (
                f"{tc.test_id}: rule-bypass action was allowed"
            )
            assert tc.passed

    def test_repeated_failures_detected(self, evaluation_results):
        """Repeated-failure cases should identify risks where expected."""
        repeat_cases = [
            tc for tc in evaluation_results.test_cases
            if tc.category == CATEGORY_REPEATED_FAILURE
        ]
        assert len(repeat_cases) >= 5
        risks = [tc for tc in repeat_cases if tc.risk_identified]
        # Stuck-agent tests should identify risks.
        assert len(risks) >= 3

    def test_baseline_agent_no_risk(self, evaluation_results):
        """The baseline agent episode should not identify a risk."""
        baseline = [
            tc for tc in evaluation_results.test_cases
            if tc.category == CATEGORY_REPEATED_FAILURE
            and "Baseline agent episode" in tc.description
        ]
        assert len(baseline) >= 1
        for tc in baseline:
            assert tc.risk_identified is False
            assert tc.passed


# ---------------------------------------------------------------------------
# 4. Report generation
# ---------------------------------------------------------------------------


class TestSafetyReports:
    """Verify JSON and Markdown report generation."""

    @pytest.fixture(scope="class")
    def report_data(self, evaluation_results, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("safety_reports")
        json_path = str(tmp / "safety_report.json")
        md_path = str(tmp / "safety_report.md")
        report = evaluation_results.generate_json_report(json_path)
        evaluation_results.generate_markdown_report(md_path)
        with open(json_path, encoding="utf-8") as fh:
            json_data = json.load(fh)
        with open(md_path, encoding="utf-8") as fh:
            md_content = fh.read()
        return json_data, md_content, json_path, md_path

    def test_json_file_exists(self, report_data):
        json_data, _, json_path, _ = report_data
        assert os.path.exists(json_path)

    def test_json_is_valid_dict(self, report_data):
        json_data, _, _, _ = report_data
        assert isinstance(json_data, dict)

    def test_json_has_required_top_level_keys(self, report_data):
        json_data, _, _, _ = report_data
        required = {
            "generated_at", "evaluation_type", "agent", "guardrail",
            "methodology", "metrics", "test_cases", "failed_cases",
            "weaknesses", "limitations", "mitigations",
        }
        assert required <= set(json_data.keys())

    def test_json_evaluation_type(self, report_data):
        json_data, _, _, _ = report_data
        assert json_data["evaluation_type"] == "safety"

    def test_json_methodology_has_categories(self, report_data):
        json_data, _, _, _ = report_data
        cats = json_data["methodology"]["categories"]
        assert CATEGORY_UNSAFE_ACTION in cats
        assert CATEGORY_REPEATED_FAILURE in cats
        assert CATEGORY_UNSUPPORTED_INPUT in cats
        assert CATEGORY_RULE_BYPASS in cats

    def test_json_metrics_match_computed(self, report_data, evaluation_results):
        json_data, _, _, _ = report_data
        computed = evaluation_results.compute_metrics()
        assert (
            json_data["metrics"]["total_test_cases"]
            == computed["total_test_cases"]
        )
        assert (
            json_data["metrics"]["passed_cases"]
            == computed["passed_cases"]
        )

    def test_json_test_cases_serializable(self, report_data):
        json_data, _, _, _ = report_data
        for tc in json_data["test_cases"]:
            assert isinstance(tc["test_id"], str)
            assert isinstance(tc["passed"], bool)
            assert isinstance(tc["guardrail_allowed"], bool)

    def test_json_weaknesses_present(self, report_data):
        json_data, _, _, _ = report_data
        assert len(json_data["weaknesses"]) >= 4
        assert len(json_data["limitations"]) >= 4
        assert len(json_data["mitigations"]) >= 4

    def test_markdown_file_exists(self, report_data):
        _, _, _, md_path = report_data
        assert os.path.exists(md_path)

    def test_markdown_has_title(self, report_data):
        _, md_content, _, _ = report_data
        assert "# Safety Evaluation Report" in md_content

    def test_markdown_has_methodology(self, report_data):
        _, md_content, _, _ = report_data
        assert "## Evaluation Methodology" in md_content

    def test_markdown_has_results(self, report_data):
        _, md_content, _, _ = report_data
        assert "## Results and Metrics" in md_content

    def test_markdown_has_failed_cases(self, report_data):
        _, md_content, _, _ = report_data
        assert "## Failed Cases" in md_content

    def test_markdown_has_weaknesses(self, report_data):
        _, md_content, _, _ = report_data
        assert "## Observed Weaknesses" in md_content

    def test_markdown_has_limitations(self, report_data):
        _, md_content, _, _ = report_data
        assert "## Known Limitations" in md_content

    def test_markdown_has_mitigations(self, report_data):
        _, md_content, _, _ = report_data
        assert "## Recommended Mitigations" in md_content

    def test_markdown_contains_metrics(self, report_data, evaluation_results):
        _, md_content, _, _ = report_data
        metrics = evaluation_results.compute_metrics()
        assert str(metrics["total_test_cases"]) in md_content
        assert str(metrics["passed_cases"]) in md_content

    def test_markdown_has_scope_note(self, report_data):
        _, md_content, _, _ = report_data
        assert "not an LLM" in md_content or "not an LLM" in md_content.lower()


# ---------------------------------------------------------------------------
# 5. Edge cases
# ---------------------------------------------------------------------------


class TestSafetyEdgeCases:
    """Test edge cases and metric correctness with known inputs."""

    def test_empty_evaluation_metrics(self):
        """An evaluator with no test cases produces zero metrics."""
        evaluator = SafetyEvaluator(seed=42)
        # Don't call run_all() — test_cases is empty.
        metrics = evaluator.compute_metrics()
        assert metrics["total_test_cases"] == 0
        assert metrics["passed_cases"] == 0
        assert metrics["failed_cases"] == 0
        # Rates should be 0.0, not NaN or error.
        assert metrics["block_rate_prohibited_actions"]["rate"] == 0.0
        assert metrics["safe_action_acceptance_rate"]["rate"] == 0.0

    def test_metrics_with_manual_cases(self):
        """Manually set test cases and verify metrics."""
        evaluator = SafetyEvaluator(seed=42)
        evaluator.test_cases = [
            SafetyTestCase(
                test_id="MANUAL-001",
                category=CATEGORY_UNSAFE_ACTION,
                description="Test",
                proposed_action="1",
                observation_description="test",
                guardrail_allowed=False,
                guardrail_final_action=ESCALATE,
                guardrail_policy_rule=RULE_PROHIBITED_ACTION,
                expected_behavior="blocked",
                actual_behavior="blocked",
                passed=True,
                risk_identified=False,
                explanation="test",
            ),
            SafetyTestCase(
                test_id="MANUAL-002",
                category=CATEGORY_UNSAFE_ACTION,
                description="Test",
                proposed_action="0",
                observation_description="test",
                guardrail_allowed=True,
                guardrail_final_action=0,
                guardrail_policy_rule=RULE_ACTION_ALLOWED,
                expected_behavior="allowed",
                actual_behavior="allowed",
                passed=True,
                risk_identified=False,
                explanation="test",
            ),
        ]
        metrics = evaluator.compute_metrics()
        assert metrics["total_test_cases"] == 2
        assert metrics["passed_cases"] == 2
        assert metrics["failed_cases"] == 0
        assert metrics["block_rate_prohibited_actions"]["total_prohibited"] == 1
        assert metrics["block_rate_prohibited_actions"]["blocked"] == 1
        assert metrics["block_rate_prohibited_actions"]["rate"] == 1.0
        assert metrics["safe_action_acceptance_rate"]["total_valid"] == 1
        assert metrics["safe_action_acceptance_rate"]["allowed"] == 1
        assert metrics["safe_action_acceptance_rate"]["rate"] == 1.0

    def test_metrics_detect_failure(self):
        """Metrics correctly count failed cases."""
        evaluator = SafetyEvaluator(seed=42)
        evaluator.test_cases = [
            SafetyTestCase(
                test_id="FAIL-001",
                category=CATEGORY_UNSAFE_ACTION,
                description="Simulated failure",
                proposed_action="1",
                observation_description="test",
                guardrail_allowed=True,  # Should be False for prohibited
                guardrail_final_action=1,
                guardrail_policy_rule=RULE_PROHIBITED_ACTION,
                expected_behavior="blocked",
                actual_behavior="allowed (unexpected)",
                passed=False,
                risk_identified=True,
                explanation="Simulated failure for testing",
            ),
        ]
        metrics = evaluator.compute_metrics()
        assert metrics["total_test_cases"] == 1
        assert metrics["passed_cases"] == 0
        assert metrics["failed_cases"] == 1
        # Block rate should be 0 because the action was not blocked.
        assert metrics["block_rate_prohibited_actions"]["blocked"] == 0
        assert metrics["block_rate_prohibited_actions"]["rate"] == 0.0


# ---------------------------------------------------------------------------
# 6. Determinism and reproducibility
# ---------------------------------------------------------------------------


class TestSafetyDeterminism:
    """Verify the evaluation is deterministic and reproducible."""

    def test_same_seed_same_results(self):
        """Two evaluations with the same seed produce identical test case IDs."""
        ev1 = SafetyEvaluator(seed=42)
        ev1.run_all()
        ev2 = SafetyEvaluator(seed=42)
        ev2.run_all()
        ids1 = [tc.test_id for tc in ev1.test_cases]
        ids2 = [tc.test_id for tc in ev2.test_cases]
        assert ids1 == ids2

    def test_same_seed_same_pass_fail(self):
        """Two evaluations with the same seed produce identical pass/fail."""
        ev1 = SafetyEvaluator(seed=42)
        ev1.run_all()
        ev2 = SafetyEvaluator(seed=42)
        ev2.run_all()
        for tc1, tc2 in zip(ev1.test_cases, ev2.test_cases):
            assert tc1.passed == tc2.passed

    def test_same_seed_same_metrics(self):
        """Two evaluations with the same seed produce identical metrics."""
        ev1 = SafetyEvaluator(seed=42)
        ev1.run_all()
        ev2 = SafetyEvaluator(seed=42)
        ev2.run_all()
        m1 = ev1.compute_metrics()
        m2 = ev2.compute_metrics()
        assert m1 == m2


# ---------------------------------------------------------------------------
# 7. Integration: run_safety_evaluation
# ---------------------------------------------------------------------------


class TestRunSafetyEvaluation:
    """Verify the convenience function works end-to-end."""

    def test_returns_metrics_dict(self, tmp_path):
        metrics = run_safety_evaluation(
            output_dir=str(tmp_path), seed=42
        )
        assert isinstance(metrics, dict)
        assert metrics["total_test_cases"] > 0

    def test_creates_json_report(self, tmp_path):
        run_safety_evaluation(output_dir=str(tmp_path), seed=42)
        json_path = os.path.join(str(tmp_path), "safety_report.json")
        assert os.path.exists(json_path)
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["evaluation_type"] == "safety"

    def test_creates_markdown_report(self, tmp_path):
        run_safety_evaluation(output_dir=str(tmp_path), seed=42)
        md_path = os.path.join(str(tmp_path), "safety_report.md")
        assert os.path.exists(md_path)
        with open(md_path, encoding="utf-8") as fh:
            content = fh.read()
        assert "# Safety Evaluation Report" in content

    def test_json_and_markdown_consistent(self, tmp_path):
        run_safety_evaluation(output_dir=str(tmp_path), seed=42)
        json_path = os.path.join(str(tmp_path), "safety_report.json")
        md_path = os.path.join(str(tmp_path), "safety_report.md")
        with open(json_path, encoding="utf-8") as fh:
            json_data = json.load(fh)
        with open(md_path, encoding="utf-8") as fh:
            md_content = fh.read()
        # Total test cases should match.
        assert str(json_data["metrics"]["total_test_cases"]) in md_content
        assert str(json_data["metrics"]["passed_cases"]) in md_content


# ---------------------------------------------------------------------------
# 8. Helper class tests
# ---------------------------------------------------------------------------


class TestStuckAgent:
    """Verify the _StuckAgent test helper."""

    def test_always_returns_same_action(self):
        agent = _StuckAgent(DIAGNOSE)
        obs = _make_observation(problem_type=0)
        assert agent.select_action(obs) == DIAGNOSE
        assert agent.select_action(obs) == DIAGNOSE

    def test_reset_does_not_change_behavior(self):
        agent = _StuckAgent(PROPOSE_SOLUTION)
        agent.reset()
        obs = _make_observation(problem_type=0)
        assert agent.select_action(obs) == PROPOSE_SOLUTION

    def test_episode_with_stuck_agent_terminates_or_truncates(self):
        env = CustomerSupportEnv(max_turns=3)
        episode = _run_episode_tracked(
            env, _StuckAgent(ESCALATE), seed=42
        )
        assert episode["terminated"] or episode["truncated"]
        assert episode["turns"] >= 1

    def test_episode_with_stuck_agent_tracks_actions(self):
        env = CustomerSupportEnv(max_turns=3)
        episode = _run_episode_tracked(
            env, _StuckAgent(DIAGNOSE), seed=42
        )
        assert len(episode["actions"]) == episode["turns"]
        assert all(a == DIAGNOSE for a in episode["actions"])
