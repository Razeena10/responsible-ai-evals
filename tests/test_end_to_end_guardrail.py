"""Tests for the end-to-end guardrail evaluation.

The central property is that a blocked original action is never executed. The
wrapper records the action it passes to ``env.step`` on every step, so the
tests can assert prevention directly rather than inferring it from rewards.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from src.baseline_agent import BaselineSupportAgent
from src.config import ESCALATE
from src.end_to_end_evaluator import (
    CONFIG_NAMES,
    EPISODES_PER_CONFIG,
    run_end_to_end_evaluation,
    run_guarded_episode,
    run_unguarded_episode,
    write_report,
)
from src.guardrail_wrapper import (
    AlwaysProhibitedAgent,
    GuardrailWrapper,
    ProhibitedProbeAgent,
)
from src.support_env import CustomerSupportEnv


@pytest.fixture(scope="module")
def report() -> dict:
    return run_end_to_end_evaluation()


# ---------------------------------------------------------------------------
# Wrapper integration
# ---------------------------------------------------------------------------


def test_blocked_action_is_replaced_before_env_step():
    env = CustomerSupportEnv(max_turns=8)
    wrapper = GuardrailWrapper(env)
    observation, _ = wrapper.reset(seed=0)
    agent = AlwaysProhibitedAgent()
    agent.reset()
    proposed = agent.select_action(observation)

    _, _, terminated, _, info = wrapper.step(proposed)
    record = wrapper.step_records[-1]

    assert record.allowed is False
    assert wrapper.blocked_actions == 1
    assert record.executed_action == record.final_action
    assert record.executed_action != record.original_action
    assert wrapper.prevention_violations == 0
    assert wrapper.invalid_actions_reaching_env == 0
    assert 0 <= info["scenario_id"] < 5
    assert terminated is True
    assert record.final_action == ESCALATE
    wrapper.close()


def test_guardrail_crash_fails_safe_at_wrapper_boundary():
    def corrupt_mask(observation):
        corrupted = dict(observation)
        corrupted["available_actions"] = np.full(7, np.nan)
        return corrupted

    env = CustomerSupportEnv(max_turns=8)
    wrapper = GuardrailWrapper(env, observation_mutator=corrupt_mask)
    wrapper.reset(seed=0)

    _, _, terminated, _, _ = wrapper.step(0)
    record = wrapper.step_records[-1]

    assert record.policy_error == "ValueError"
    assert record.allowed is False
    assert record.policy_rule == "EXCEPTION:ValueError"
    assert record.executed_action == ESCALATE
    assert wrapper.policy_errors == 1
    assert wrapper.prevention_violations == 0
    assert terminated is True
    wrapper.close()


def test_guarded_episode_matches_unguarded_for_baseline_agent():
    for seed in (20260101, 20260102, 20260103, 20260104, 20260105):
        unguarded = run_unguarded_episode(BaselineSupportAgent(), seed)
        guarded = run_guarded_episode(BaselineSupportAgent(), seed)
        assert guarded["success"] == unguarded["success"]
        assert guarded["reward"] == pytest.approx(unguarded["reward"])
        assert guarded["turns"] == unguarded["turns"]
        assert guarded["blocked"] == 0


def test_prohibited_probe_is_blocked_and_never_reaches_env():
    guarded = run_guarded_episode(ProhibitedProbeAgent(), 20260101)
    unguarded = run_unguarded_episode(ProhibitedProbeAgent(), 20260101)
    assert unguarded["prohibited_actions_attempted"] >= 1
    assert unguarded["prohibited_actions_reaching_env"] >= 1
    assert guarded["prohibited_actions_attempted"] >= 1
    assert guarded["prohibited_actions_reaching_env"] == 0
    assert guarded["prevention_violations"] == 0


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def test_report_shape(report):
    assert report["metadata"]["episodes_per_config"] == EPISODES_PER_CONFIG
    assert set(report["configs"]) == set(CONFIG_NAMES)
    seeds = report["metadata"]["seeds"]["values"]
    assert len(seeds) == EPISODES_PER_CONFIG
    assert len(set(seeds)) == EPISODES_PER_CONFIG
    for name in CONFIG_NAMES:
        assert report["configs"][name]["metrics"]["episodes"] == (
            EPISODES_PER_CONFIG
        )


def test_guarded_baseline_has_no_effect_on_this_agent(report):
    unguarded = report["configs"]["baseline_unguarded"]["metrics"]
    guarded = report["configs"]["baseline_guarded"]["metrics"]
    assert guarded["successes"] == unguarded["successes"]
    assert guarded["avg_reward"] == pytest.approx(unguarded["avg_reward"])
    assert guarded["avg_turns"] == pytest.approx(unguarded["avg_turns"])
    assert guarded["blocked_actions"] == 0
    assert report["deltas"]["guarded_minus_unguarded"]["success_rate"] == 0.0


def test_probe_prevention_is_complete_in_aggregate(report):
    unguarded = report["configs"]["prohibited_probe_unguarded"]["metrics"]
    guarded = report["configs"]["prohibited_probe_guarded"]["metrics"]
    assert unguarded["prohibited_actions_attempted"] > 0
    assert (
        unguarded["prohibited_actions_reaching_env"]
        == unguarded["prohibited_actions_attempted"]
    )
    assert guarded["prohibited_actions_reaching_env"] == 0
    assert guarded["blocked_actions"] == guarded[
        "prohibited_actions_attempted"
    ]
    assert guarded["fallback_escalations"] == guarded["blocked_actions"]
    assert guarded["episodes_terminated_by_guardrail"] == (
        guarded["episodes"]
    )


def test_integrity_checks(report):
    integrity = report["integrity_checks"]
    assert integrity["blocked_actions_never_reach_env"] is True
    for name, count in integrity["invalid_actions_reaching_env"].items():
        assert count == 0, name
    for name in CONFIG_NAMES:
        metrics = report["configs"][name]["metrics"]
        assert metrics["prevention_violations"] == 0
        assert metrics["policy_errors"] == 0


def test_corrupted_observation_config_blocks_every_step(report):
    metrics = report["configs"]["baseline_corrupted_observation"]["metrics"]
    assert metrics["blocked_actions"] == metrics["episodes"]
    assert metrics["episodes_terminated_by_guardrail"] == metrics["episodes"]
    assert metrics["successes"] == 0
    assert metrics["invalid_actions_reaching_env"] == 0


def test_json_report_written(report, tmp_path):
    path = write_report(report, str(tmp_path))
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    assert loaded["integrity_checks"] == report["integrity_checks"]
    assert set(loaded["configs"]) == set(CONFIG_NAMES)
