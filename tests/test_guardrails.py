"""Tests for the guardrails module.

Tests cover:
    1. Valid actions are allowed.
    2. Invalid action IDs are blocked.
    3. Prohibited actions are blocked.
    4. Malformed observations are handled safely.
    5. Safe fallback actions are valid.
    6. The decision result contains the expected information.
    7. Normal safe support actions still work (integration with agent + env).
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.guardrails import (
    GuardrailPolicy,
    GuardrailResult,
    evaluate_action,
    guard,
    RULE_ACTION_ALLOWED,
    RULE_INVALID_ACTION_ID,
    RULE_PROHIBITED_ACTION,
    RULE_MALFORMED_OBSERVATION,
)
from src.support_scenarios import SCENARIOS, NUM_SCENARIOS
from src.baseline_agent import BaselineSupportAgent
from src.support_env import CustomerSupportEnv


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_observation(
    problem_type: int = 0,
    collected: np.ndarray | None = None,
    frustration: float = 0.0,
    previous_response: int = 0,
    turn_count: int = 0,
) -> dict:
    """Build a valid observation dict matching the environment format.

    Args:
        problem_type: Scenario ID (0-4).
        collected: Binary array of collected info (length MAX_INFO_FIELDS).
        frustration: Frustration level [0.0, 1.0].
        previous_response: User's last message type.
        turn_count: Current turn number.

    Returns:
        A dict matching the environment's observation format.
    """
    scenario = SCENARIOS[problem_type]
    if collected is None:
        collected = np.zeros(config.MAX_INFO_FIELDS, dtype=np.int8)
    missing = (1 - collected).astype(np.int8)
    available = np.zeros(config.NUM_ACTIONS, dtype=np.int8)
    for action_id in scenario.available_actions:
        available[action_id] = 1
    return {
        "customer_issue": np.int64(problem_type),
        "collected_information": collected,
        "missing_information": missing,
        "turn_count": np.int64(turn_count),
        "frustration": np.array(frustration, dtype=np.float32),
        "previous_customer_response": np.int64(previous_response),
        "available_actions": available,
        "problem_type": np.int64(problem_type),
    }


def _all_collected(problem_type: int) -> np.ndarray:
    """Build a collected_information array with all required fields set."""
    scenario = SCENARIOS[problem_type]
    collected = np.zeros(config.MAX_INFO_FIELDS, dtype=np.int8)
    for i in range(len(scenario.required_fields)):
        collected[i] = 1
    return collected


# ---------------------------------------------------------------------------
# 1. Valid actions are allowed
# ---------------------------------------------------------------------------


class TestValidActionsAllowed:
    """Valid actions that are available for the current scenario are allowed."""

    def test_ask_account_info_allowed(self):
        """ASK_ACCOUNT_INFO is available in scenario 0 and should be allowed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert result.allowed is True
        assert result.final_action == config.ASK_ACCOUNT_INFO
        assert result.policy_rule == RULE_ACTION_ALLOWED

    def test_diagnose_allowed(self):
        """DIAGNOSE is available in scenario 0 and should be allowed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0, collected=_all_collected(0))
        result = policy.evaluate(config.DIAGNOSE, obs)
        assert result.allowed is True
        assert result.final_action == config.DIAGNOSE

    def test_propose_solution_allowed(self):
        """PROPOSE_SOLUTION is available in scenario 0 and should be allowed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0, collected=_all_collected(0))
        result = policy.evaluate(config.PROPOSE_SOLUTION, obs)
        assert result.allowed is True
        assert result.final_action == config.PROPOSE_SOLUTION

    def test_escalate_allowed(self):
        """ESCALATE is available in all scenarios and should be allowed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.ESCALATE, obs)
        assert result.allowed is True
        assert result.final_action == config.ESCALATE

    def test_apologize_allowed(self):
        """APOLOGIZE is available in all scenarios and should be allowed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.APOLOGIZE, obs)
        assert result.allowed is True
        assert result.final_action == config.APOLOGIZE

    def test_ask_billing_info_allowed_in_billing_scenario(self):
        """ASK_BILLING_INFO is available in scenario 4 (Billing Issue)."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=4)
        result = policy.evaluate(config.ASK_BILLING_INFO, obs)
        assert result.allowed is True
        assert result.final_action == config.ASK_BILLING_INFO

    def test_ask_technical_info_allowed_in_tech_scenario(self):
        """ASK_TECHNICAL_INFO is available in scenario 0 (Internet Connection)."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.ASK_TECHNICAL_INFO, obs)
        assert result.allowed is True
        assert result.final_action == config.ASK_TECHNICAL_INFO

    @pytest.mark.parametrize(
        "scenario_idx", range(NUM_SCENARIOS), ids=lambda s: SCENARIOS[s].name
    )
    def test_all_available_actions_allowed(self, scenario_idx: int):
        """Every action listed in the scenario's available_actions is allowed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=scenario_idx)
        scenario = SCENARIOS[scenario_idx]
        for action_id in scenario.available_actions:
            result = policy.evaluate(action_id, obs)
            assert result.allowed is True, (
                f"Scenario '{scenario.name}': action "
                f"{config.ACTION_NAMES[action_id]} ({action_id}) should be "
                f"allowed but was blocked: {result.reason}"
            )
            assert result.final_action == action_id

    def test_module_level_evaluate_action(self):
        """The module-level evaluate_action() convenience function works."""
        obs = _make_observation(problem_type=0)
        result = evaluate_action(config.ASK_ACCOUNT_INFO, obs)
        assert result.allowed is True
        assert result.final_action == config.ASK_ACCOUNT_INFO

    def test_numpy_integer_action_allowed(self):
        """numpy integer types are accepted as valid actions."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(np.int64(config.ASK_ACCOUNT_INFO), obs)
        assert result.allowed is True
        assert result.final_action == config.ASK_ACCOUNT_INFO


# ---------------------------------------------------------------------------
# 2. Invalid action IDs are blocked
# ---------------------------------------------------------------------------


class TestInvalidActionIDsBlocked:
    """Actions that are not valid integers in [0, NUM_ACTIONS) are blocked."""

    def test_negative_action_blocked(self):
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(-1, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_INVALID_ACTION_ID
        assert result.original_action == -1

    def test_too_large_action_blocked(self):
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(99, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_INVALID_ACTION_ID
        assert result.original_action == 99

    def test_boundary_action_blocked(self):
        """Action == NUM_ACTIONS is one past the max and must be blocked."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.NUM_ACTIONS, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_INVALID_ACTION_ID

    def test_float_action_blocked(self):
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(3.5, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_INVALID_ACTION_ID
        assert result.original_action is None

    def test_string_action_blocked(self):
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate("ASK_ACCOUNT_INFO", obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_INVALID_ACTION_ID
        assert result.original_action is None

    def test_none_action_blocked(self):
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(None, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_INVALID_ACTION_ID
        assert result.original_action is None

    def test_bool_action_blocked(self):
        """bool is a subclass of int but must be rejected as an action."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(True, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_INVALID_ACTION_ID
        assert result.original_action is None

    def test_list_action_blocked(self):
        """A list is not a valid action."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate([0], obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_INVALID_ACTION_ID

    def test_numpy_float_action_blocked(self):
        """numpy float types are not valid actions."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(np.float64(3.0), obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_INVALID_ACTION_ID


# ---------------------------------------------------------------------------
# 3. Prohibited actions are blocked
# ---------------------------------------------------------------------------


class TestProhibitedActionsBlocked:
    """Valid action IDs that are not available for the current scenario are blocked."""

    def test_billing_info_blocked_in_scenario_0(self):
        """ASK_BILLING_INFO is not available in scenario 0 (Internet Connection)."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.ASK_BILLING_INFO, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_PROHIBITED_ACTION
        assert result.original_action == config.ASK_BILLING_INFO

    def test_technical_info_blocked_in_scenario_3(self):
        """ASK_TECHNICAL_INFO is not available in scenario 3 (Password Reset)."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=3)
        result = policy.evaluate(config.ASK_TECHNICAL_INFO, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_PROHIBITED_ACTION

    def test_billing_info_blocked_in_scenario_3(self):
        """ASK_BILLING_INFO is not available in scenario 3 (Password Reset)."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=3)
        result = policy.evaluate(config.ASK_BILLING_INFO, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_PROHIBITED_ACTION

    def test_technical_info_blocked_in_scenario_4(self):
        """ASK_TECHNICAL_INFO is not available in scenario 4 (Billing Issue)."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=4)
        result = policy.evaluate(config.ASK_TECHNICAL_INFO, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_PROHIBITED_ACTION

    @pytest.mark.parametrize(
        "scenario_idx", range(NUM_SCENARIOS), ids=lambda s: SCENARIOS[s].name
    )
    def test_prohibited_actions_blocked_all_scenarios(self, scenario_idx: int):
        """Every action NOT in the scenario's available_actions is blocked."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=scenario_idx)
        scenario = SCENARIOS[scenario_idx]
        all_actions = set(range(config.NUM_ACTIONS))
        prohibited = all_actions - set(scenario.available_actions)
        for action_id in prohibited:
            result = policy.evaluate(action_id, obs)
            assert result.allowed is False, (
                f"Scenario '{scenario.name}': action "
                f"{config.ACTION_NAMES[action_id]} ({action_id}) should be "
                f"prohibited but was allowed"
            )
            assert result.final_action == config.ESCALATE
            assert result.policy_rule == RULE_PROHIBITED_ACTION

    def test_reason_mentions_scenario_name(self):
        """The reason string includes the scenario name for debugging."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.ASK_BILLING_INFO, obs)
        assert "Internet Connection Problem" in result.reason


# ---------------------------------------------------------------------------
# 4. Malformed observations are handled safely
# ---------------------------------------------------------------------------


class TestMalformedObservationsHandled:
    """Malformed observations do not crash; the action is blocked safely."""

    def test_none_observation(self):
        policy = GuardrailPolicy()
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, None)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_non_dict_observation(self):
        """A string is not a valid observation."""
        policy = GuardrailPolicy()
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, "not a dict")
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_empty_dict_observation(self):
        """An empty dict is missing all required keys."""
        policy = GuardrailPolicy()
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, {})
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_missing_available_actions_key(self):
        """Observation missing the available_actions key is malformed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        del obs["available_actions"]
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_missing_problem_type_key(self):
        """Observation missing the problem_type key is malformed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        del obs["problem_type"]
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_wrong_length_available_actions(self):
        """available_actions with wrong length is malformed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        obs["available_actions"] = np.zeros(5, dtype=np.int8)
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_none_available_actions(self):
        """available_actions set to None is malformed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        obs["available_actions"] = None
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_invalid_problem_type_out_of_range(self):
        """problem_type outside the valid range is malformed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        obs["problem_type"] = np.int64(99)
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_non_integer_problem_type(self):
        """problem_type as a string is malformed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        obs["problem_type"] = "zero"
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_none_problem_type(self):
        """problem_type set to None is malformed."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        obs["problem_type"] = None
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_malformed_observation_with_invalid_action(self):
        """Both malformed observation and invalid action: observation checked first."""
        policy = GuardrailPolicy()
        result = policy.evaluate(99, None)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_does_not_raise_on_malformed_observation(self):
        """No exception is raised for any malformed observation input."""
        policy = GuardrailPolicy()
        malformed_inputs = [
            None,
            {},
            "string",
            42,
            [],
            {"available_actions": np.zeros(config.NUM_ACTIONS, dtype=np.int8)},
        ]
        for obs in malformed_inputs:
            result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
            assert result.allowed is False
            assert result.final_action == config.ESCALATE


# ---------------------------------------------------------------------------
# 5. Safe fallback actions are valid
# ---------------------------------------------------------------------------


class TestSafeFallbackValid:
    """When an action is blocked, the fallback action is always valid."""

    def test_default_fallback_is_escalate(self):
        """The default fallback action is ESCALATE."""
        policy = GuardrailPolicy()
        assert policy.fallback_action == config.ESCALATE

    def test_fallback_is_valid_action_id(self):
        """The fallback action is in [0, NUM_ACTIONS)."""
        policy = GuardrailPolicy()
        assert 0 <= policy.fallback_action < config.NUM_ACTIONS

    def test_fallback_available_in_all_scenarios(self):
        """ESCALATE is available in all five scenarios."""
        for scenario in SCENARIOS:
            assert config.ESCALATE in scenario.available_actions, (
                f"ESCALATE not available in scenario '{scenario.name}'"
            )

    def test_blocked_returns_fallback(self):
        """Every blocked decision returns the fallback as final_action."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)

        # Invalid action
        result = policy.evaluate(99, obs)
        assert result.final_action == config.ESCALATE

        # Prohibited action
        result = policy.evaluate(config.ASK_BILLING_INFO, obs)
        assert result.final_action == config.ESCALATE

        # Malformed observation
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, {})
        assert result.final_action == config.ESCALATE

    def test_custom_fallback_action(self):
        """A custom fallback action can be specified."""
        policy = GuardrailPolicy(fallback_action=config.APOLOGIZE)
        assert policy.fallback_action == config.APOLOGIZE
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(99, obs)
        assert result.final_action == config.APOLOGIZE

    def test_invalid_fallback_action_raises(self):
        """An invalid fallback action raises ValueError at construction."""
        with pytest.raises(ValueError, match="fallback_action"):
            GuardrailPolicy(fallback_action=99)
        with pytest.raises(ValueError, match="fallback_action"):
            GuardrailPolicy(fallback_action=-1)
        with pytest.raises(ValueError, match="fallback_action"):
            GuardrailPolicy(fallback_action="ESCALATE")

    def test_guard_returns_valid_action(self):
        """The guard() convenience function always returns a valid action."""
        obs = _make_observation(problem_type=0)
        assert guard(config.ASK_ACCOUNT_INFO, obs) == config.ASK_ACCOUNT_INFO
        assert guard(99, obs) == config.ESCALATE
        assert guard(config.ASK_BILLING_INFO, obs) == config.ESCALATE
        assert guard(config.ASK_ACCOUNT_INFO, None) == config.ESCALATE


# ---------------------------------------------------------------------------
# 6. Decision result contains expected information
# ---------------------------------------------------------------------------


class TestDecisionResultContainsInfo:
    """The GuardrailResult contains all required fields with correct types."""

    def test_result_is_guardrail_result(self):
        """evaluate() returns a GuardrailResult instance."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert isinstance(result, GuardrailResult)

    def test_allowed_result_has_all_fields(self):
        """An allowed result has all five fields populated."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert result.original_action == config.ASK_ACCOUNT_INFO
        assert result.final_action == config.ASK_ACCOUNT_INFO
        assert result.allowed is True
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0
        assert result.policy_rule == RULE_ACTION_ALLOWED

    def test_blocked_invalid_result_has_all_fields(self):
        """A blocked (invalid action) result has all five fields."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(99, obs)
        assert result.original_action == 99
        assert result.final_action == config.ESCALATE
        assert result.allowed is False
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0
        assert result.policy_rule == RULE_INVALID_ACTION_ID

    def test_blocked_prohibited_result_has_all_fields(self):
        """A blocked (prohibited action) result has all five fields."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.ASK_BILLING_INFO, obs)
        assert result.original_action == config.ASK_BILLING_INFO
        assert result.final_action == config.ESCALATE
        assert result.allowed is False
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0
        assert result.policy_rule == RULE_PROHIBITED_ACTION

    def test_blocked_malformed_result_has_all_fields(self):
        """A blocked (malformed observation) result has all five fields."""
        policy = GuardrailPolicy()
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, None)
        assert result.final_action == config.ESCALATE
        assert result.allowed is False
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0
        assert result.policy_rule == RULE_MALFORMED_OBSERVATION

    def test_field_types(self):
        """All fields have the expected types."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert isinstance(result.original_action, (int, type(None)))
        assert isinstance(result.final_action, int)
        assert isinstance(result.allowed, bool)
        assert isinstance(result.reason, str)
        assert isinstance(result.policy_rule, str)

    def test_original_action_none_for_non_integer(self):
        """original_action is None when the action is not an integer."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate("hello", obs)
        assert result.original_action is None

    def test_original_action_int_for_out_of_range(self):
        """original_action records the integer even if out of range."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(99, obs)
        assert result.original_action == 99

    def test_result_is_frozen(self):
        """GuardrailResult is immutable (frozen dataclass)."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        result = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        with pytest.raises(Exception):
            result.allowed = False  # type: ignore[misc]

    def test_reason_is_informative(self):
        """The reason string is non-empty and describes the decision."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)

        allowed = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert "allowed" in allowed.reason.lower()

        blocked = policy.evaluate(config.ASK_BILLING_INFO, obs)
        assert "not available" in blocked.reason.lower()
        assert "ASK_BILLING_INFO" in blocked.reason


# ---------------------------------------------------------------------------
# 7. Normal safe support actions still work
# ---------------------------------------------------------------------------


class TestNormalSupportActionsWork:
    """The guardrail does not interfere with normal agent operations."""

    def test_guardrail_allows_all_agent_actions(self):
        """Every action the baseline agent selects passes the guardrail."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        policy = GuardrailPolicy()
        obs, info = env.reset(seed=42)
        agent.reset()

        for _ in range(20):
            action = agent.select_action(obs)
            result = policy.evaluate(action, obs)
            assert result.allowed is True, (
                f"Agent action {config.ACTION_NAMES.get(action, action)} "
                f"was blocked by guardrail: {result.reason}"
            )
            obs, _, terminated, truncated, _ = env.step(result.final_action)
            if terminated or truncated:
                break

    def test_full_episode_with_guardrail(self):
        """A full episode runs normally with the guardrail in the loop."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        policy = GuardrailPolicy()
        obs, _ = env.reset(seed=42)
        agent.reset()

        turns = 0
        terminated = False
        truncated = False
        while not terminated and not truncated and turns < 50:
            action = agent.select_action(obs)
            guarded_action = policy.guard(action, obs)
            assert 0 <= guarded_action < config.NUM_ACTIONS
            obs, _, terminated, truncated, _ = env.step(guarded_action)
            turns += 1

        assert terminated or truncated
        assert turns > 0

    def test_guardrail_does_not_change_agent_action(self):
        """When allowed, final_action equals the original action."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        policy = GuardrailPolicy()
        obs, _ = env.reset(seed=42)
        agent.reset()

        for _ in range(20):
            action = agent.select_action(obs)
            result = policy.evaluate(action, obs)
            assert result.final_action == action
            obs, _, terminated, truncated, _ = env.step(result.final_action)
            if terminated or truncated:
                break

    def test_multiple_episodes_with_guardrail(self):
        """The guardrail works across multiple episodes without issues."""
        env = CustomerSupportEnv()
        agent = BaselineSupportAgent()
        policy = GuardrailPolicy()

        for seed in range(42, 52):
            obs, _ = env.reset(seed=seed)
            agent.reset()
            turns = 0
            terminated = False
            truncated = False
            while not terminated and not truncated and turns < 50:
                action = agent.select_action(obs)
                guarded_action = policy.guard(action, obs)
                assert 0 <= guarded_action < config.NUM_ACTIONS
                obs, _, terminated, truncated, _ = env.step(guarded_action)
                turns += 1
            assert terminated or truncated, (
                f"Episode with seed {seed} did not terminate"
            )

    def test_guardrail_with_real_env_observation(self):
        """The guardrail works with observations from the actual environment."""
        env = CustomerSupportEnv()
        policy = GuardrailPolicy()
        obs, info = env.reset(seed=42)
        scenario = SCENARIOS[info["scenario_id"]]

        # Test an available action
        available_action = scenario.available_actions[0]
        result = policy.evaluate(available_action, obs)
        assert result.allowed is True

        # Test a prohibited action (find one not in available_actions)
        all_actions = set(range(config.NUM_ACTIONS))
        prohibited = all_actions - set(scenario.available_actions)
        if prohibited:
            prohibited_action = next(iter(prohibited))
            result = policy.evaluate(prohibited_action, obs)
            assert result.allowed is False
            assert result.final_action == config.ESCALATE

    def test_guardrail_catches_injected_invalid_action(self):
        """If an invalid action is injected, the guardrail blocks it and
        the safe fallback (ESCALATE) can still be stepped through the env."""
        env = CustomerSupportEnv()
        policy = GuardrailPolicy()
        obs, _ = env.reset(seed=42)

        # Inject an invalid action
        result = policy.evaluate(999, obs)
        assert result.allowed is False
        assert result.final_action == config.ESCALATE

        # The fallback (ESCALATE) is a valid env action
        obs, _, terminated, truncated, _ = env.step(result.final_action)
        assert terminated is True  # ESCALATE terminates the episode

    def test_deterministic_decisions(self):
        """The same inputs always produce the same decision."""
        policy = GuardrailPolicy()
        obs = _make_observation(problem_type=0)
        r1 = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        r2 = policy.evaluate(config.ASK_ACCOUNT_INFO, obs)
        assert r1 == r2

        r3 = policy.evaluate(config.ASK_BILLING_INFO, obs)
        r4 = policy.evaluate(config.ASK_BILLING_INFO, obs)
        assert r3 == r4
