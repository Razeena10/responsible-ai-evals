"""Tests for the enhanced (robust) safety evaluation.

These tests check the evaluation harness itself: reproducibility, that
false-positive/false-negative rates are computed from the independent oracle
rather than the guardrail's own labels, boundary handling, the documented
NaN-mask robustness defect, multi-turn behaviour, and report consistency.

They deliberately assert the *observed* behaviour of the unchanged guardrail,
including its weaknesses, so a future change to the guardrail that fixes a
weakness will fail the corresponding test and force the report to be updated.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from src.config import NUM_ACTIONS
from src.guardrails import RULE_INVALID_ACTION_ID, GuardrailPolicy
from src.robust_safety_evaluator import (
    CATEGORY_LEGITIMATE_SWEEP,
    CATEGORY_MALFORMED,
    CATEGORY_REPEATED_BLOCK,
    RobustSafetyEvaluator,
    make_observation,
)
from src.safety_oracle import expected_decision


@pytest.fixture(scope="module")
def evaluator() -> RobustSafetyEvaluator:
    instance = RobustSafetyEvaluator(seed=42)
    instance.run_all()
    return instance


def _fingerprint(cases) -> list[tuple[str, str, str, str]]:
    return [
        (case.case_id, case.action, case.observation_signature, case.outcome)
        for case in cases
    ]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_every_case(evaluator):
    second = RobustSafetyEvaluator(seed=42)
    second.run_all()
    assert _fingerprint(second.cases) == _fingerprint(evaluator.cases)


def test_metrics_reproducible(evaluator):
    second = RobustSafetyEvaluator(seed=42)
    second.run_all()
    assert second.compute_metrics() == evaluator.compute_metrics()


def test_different_seed_changes_only_generated_cases(evaluator):
    other = RobustSafetyEvaluator(seed=43)
    other.run_all()
    hand_authored = {
        (case.case_id, case.action, case.observation_signature, case.outcome)
        for case in evaluator.cases
        if case.provenance == "hand_authored"
    }
    same_in_other = {
        (case.case_id, case.action, case.observation_signature, case.outcome)
        for case in other.cases
        if case.provenance == "hand_authored"
    }
    assert hand_authored == same_in_other
    assert _fingerprint(other.cases) != _fingerprint(evaluator.cases)


def test_seed_is_recorded(evaluator):
    report = evaluator.build_report()
    assert report["seed"] == 42
    generated_seeds = {
        case.seed for case in evaluator.cases if case.seed is not None
    }
    assert 42 in generated_seeds  # randomized observations
    assert len(generated_seeds) > 1  # multi-turn episode seeds are recorded


def test_minimum_generated_case_count(evaluator):
    metrics = evaluator.compute_metrics()
    assert metrics["generated_cases"] >= 300
    assert metrics["generated_cases"] < metrics["total_cases"]


# ---------------------------------------------------------------------------
# Independent metric calculations
# ---------------------------------------------------------------------------


def test_false_positive_rate_matches_case_level_recount(evaluator):
    metrics = evaluator.compute_metrics()
    legitimate = [
        case
        for case in evaluator.cases
        if case.expected_class == "legitimate_action"
    ]
    blocked = sum(
        1 for case in legitimate if case.guardrail_decision == "block"
    )
    assert len(legitimate) == metrics["legitimate_action_tests"]["total"]
    assert blocked == metrics["legitimate_action_tests"]["incorrectly_blocked"]
    rate = blocked / len(legitimate)
    assert rate == pytest.approx(
        metrics["legitimate_action_tests"]["false_positive_rate"]
    )


def test_false_negative_rate_matches_case_level_recount(evaluator):
    metrics = evaluator.compute_metrics()
    prohibited = [
        case
        for case in evaluator.cases
        if case.expected_class == "prohibited_action"
    ]
    allowed = sum(
        1 for case in prohibited if case.guardrail_decision == "allow"
    )
    assert len(prohibited) == metrics["prohibited_action_tests"]["total"]
    assert allowed == metrics["prohibited_action_tests"][
        "incorrectly_allowed"
    ]
    assert allowed / len(prohibited) == pytest.approx(
        metrics["prohibited_action_tests"]["false_negative_rate"]
    )


def test_expected_classes_come_from_independent_oracle(evaluator):
    """Sweep cases must carry the oracle's class, not the guardrail rule."""
    sweep = [
        case
        for case in evaluator.cases
        if case.category == CATEGORY_LEGITIMATE_SWEEP
    ]
    assert sweep
    for case in sweep:
        scenario = int(case.description.split("scenario ", 1)[1][0])
        action = int(case.action.split(":", 1)[1])
        _, expected_class = expected_decision(
            action, make_observation(scenario)
        )
        assert case.expected_class == expected_class
        assert expected_class == "legitimate_action"


def test_oracle_and_guardrail_are_separate_modules():
    """The oracle must not import or reference the guardrail implementation."""
    import ast

    import src.safety_oracle as oracle

    source = open(oracle.__file__, encoding="utf-8").read()
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
    assert not any("guardrail" in name for name in imported_modules)
    code_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert "GuardrailPolicy" not in code_names


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action, expected_allowed, expected_rule",
    [
        (0, True, "ACTION_ALLOWED"),
        (NUM_ACTIONS - 1, True, "ACTION_ALLOWED"),
        (NUM_ACTIONS, False, RULE_INVALID_ACTION_ID),
        (-1, False, RULE_INVALID_ACTION_ID),
        (2**31 - 1, False, RULE_INVALID_ACTION_ID),
        (True, False, RULE_INVALID_ACTION_ID),
        ("3", False, RULE_INVALID_ACTION_ID),
        (3.0, False, RULE_INVALID_ACTION_ID),
    ],
)
def test_action_boundary_decisions(
    action, expected_allowed, expected_rule
):
    result = GuardrailPolicy().evaluate(action, make_observation(0))
    assert result.allowed is expected_allowed
    assert result.policy_rule == expected_rule


def test_legitimate_sweep_covers_every_allowed_pair(evaluator):
    coverage = evaluator.coverage()
    assert coverage["distinct_decision_pairs"]["legitimate"] == 29
    assert coverage["distinct_decision_pairs"]["prohibited"] == 6
    sweep = [
        case
        for case in evaluator.cases
        if case.category == CATEGORY_LEGITIMATE_SWEEP
    ]
    assert len(sweep) == 29
    assert all(case.guardrail_decision == "allow" for case in sweep)
    assert all(case.expected_class == "legitimate_action" for case in sweep)


# ---------------------------------------------------------------------------
# Discovered malformed-input behaviour
# ---------------------------------------------------------------------------


def test_nan_mask_makes_guardrail_raise(evaluator):
    """Documented weakness: NaN mask raises instead of failing safe."""
    observation = make_observation(0)
    observation["available_actions"] = np.full(
        NUM_ACTIONS, np.nan, dtype=np.float64
    )
    with pytest.raises(ValueError):
        GuardrailPolicy().evaluate(0, observation)

    cases = [
        case
        for case in evaluator.cases
        if case.category == CATEGORY_MALFORMED
        and "available_actions contains NaN" in case.description
    ]
    assert cases
    assert all(case.guardrail_decision == "error" for case in cases)
    assert evaluator.compute_metrics()["guardrail_exceptions"] == len(cases)


def test_malformed_observation_acceptance_is_reported(evaluator):
    metrics = evaluator.compute_metrics()
    malformed = metrics["malformed_observation_tests"]
    assert malformed["total"] > 0
    assert malformed["accepted"] > 0
    assert malformed["accepted"] + malformed["blocked"] + malformed[
        "guardrail_errors"
    ] == malformed["total"]


def test_rejected_legitimate_action_under_corrupt_observation_is_not_fp(
    evaluator,
):
    """A legitimate action blocked under a corrupt mask is malformed, not FP."""
    from src.robust_safety_evaluator import CATEGORY_FALLBACK

    corrupted = [
        case
        for case in evaluator.cases
        if case.category == CATEGORY_FALLBACK
        and case.expected_class == "malformed_observation"
        and case.guardrail_decision == "block"
    ]
    assert corrupted
    metrics = evaluator.compute_metrics()
    assert metrics["legitimate_action_tests"]["incorrectly_blocked"] == 0
    assert all(
        case.expected_class != "legitimate_action" for case in corrupted
    )


# ---------------------------------------------------------------------------
# Multi-turn and repeated sequences
# ---------------------------------------------------------------------------


def test_repeated_blocked_sequence_records_are_consistent(evaluator):
    repeated = [
        case
        for case in evaluator.cases
        if case.category == CATEGORY_REPEATED_BLOCK
    ]
    assert len(repeated) == 5
    for case in repeated:
        assert "attempts=8" in case.notes
        assert "consistent=True" in case.notes
        assert case.outcome == "correct_block"


def test_multi_turn_sequences_never_execute_a_blocked_action(evaluator):
    summaries = evaluator._multi_turn_summaries
    assert len(summaries) == 2
    for summary in summaries:
        assert summary["prevention_violations"] == 0
        assert summary["invalid_actions_reaching_env"] == 0
        assert summary["blocked_actions"] == summary["steps"]
    by_config = {s["configuration"]: s for s in summaries}
    assert by_config["default_fallback_escalate"][
        "episodes_terminated_due_to_guardrail"
    ] == by_config["default_fallback_escalate"]["episodes"]
    assert by_config["non_terminating_fallback_apologize"]["steps"] > (
        by_config["non_terminating_fallback_apologize"]["episodes"]
    )


# ---------------------------------------------------------------------------
# Report consistency
# ---------------------------------------------------------------------------


def test_report_internal_consistency(evaluator):
    report = evaluator.build_report()
    metrics = report["metrics"]
    assert metrics["total_cases"] == len(evaluator.cases)
    assert metrics["hand_authored_cases"] + metrics["generated_cases"] == (
        metrics["total_cases"]
    )
    assert metrics["passed_cases"] + metrics["failed_cases"] == (
        metrics["total_cases"]
    )
    assert sum(
        block["cases"] for block in metrics["by_category"].values()
    ) == metrics["total_cases"]
    assert sum(report["failures_by_kind"].values()) == (
        metrics["failed_cases"]
    )


def test_reports_written_and_well_formed(evaluator, tmp_path):
    report = evaluator.build_report()
    evaluator.write_reports(str(tmp_path))
    json_path = os.path.join(tmp_path, "robust_safety_report.json")
    md_path = os.path.join(tmp_path, "robust_safety_report.md")
    assert os.path.exists(json_path)
    assert os.path.exists(md_path)
    with open(json_path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["metrics"]["total_cases"] == report["metrics"][
        "total_cases"
    ]
    with open(md_path, encoding="utf-8") as handle:
        markdown = handle.read()
    assert "False Positives and False Negatives" in markdown
    assert "Discovered Weaknesses" in markdown
    assert "False positives" in markdown
