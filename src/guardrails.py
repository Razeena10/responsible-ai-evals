"""Deterministic safety guardrail layer for the customer support agent.

This module implements a guardrail policy that evaluates agent actions before
they are passed to the environment. It validates action IDs, checks action
availability against the current observation and scenario definitions, and
handles malformed observations safely with a deterministic fallback to ESCALATE.

Design assumptions (documented per project requirements):
    - The agent is a deterministic, discrete-action support agent (not an LLM).
    - This is NOT a prompt-injection defense. It validates the action space.
    - The guardrail does not modify the original Week 3 agent or environment.
    - The guardrail uses only the observation dict and the proposed action.
    - ESCALATE (action 5) is available in all five scenarios, making it a safe
      universal fallback when an action is blocked.
    - No LLM, external API, network request, or additional dependency is used.
    - The guardrail is deterministic: the same inputs always produce the same
      decision.

Guardrail decision flow:
    1. Validate the observation structure (keys, types, shapes).
       -> If malformed: block, fallback to ESCALATE, record reason.
    2. Validate the action ID (integer in [0, NUM_ACTIONS)).
       -> If invalid: block, fallback to ESCALATE, record reason.
    3. Check action availability against the scenario definition
       (via problem_type -> SCENARIOS[problem_type].available_actions)
       and the observation's available_actions mask (defense in depth).
       -> If prohibited: block, fallback to ESCALATE, record reason.
    4. If all checks pass: allow the action, record reason.

Example::

    policy = GuardrailPolicy()
    result = policy.evaluate(action, observation)
    if result.allowed:
        env.step(result.final_action)
    else:
        logger.warning(result.reason)
        env.step(result.final_action)  # safe fallback (ESCALATE)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import (
    NUM_ACTIONS,
    ESCALATE,
    ACTION_NAMES,
)
from src.support_scenarios import SCENARIOS, NUM_SCENARIOS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Required observation keys — must match the environment's observation space.
# This mirrors BaselineSupportAgent._OBSERVATION_KEYS but is defined
# independently so the guardrail does not depend on the agent module.
_REQUIRED_OBSERVATION_KEYS: frozenset[str] = frozenset({
    "customer_issue",
    "collected_information",
    "missing_information",
    "turn_count",
    "frustration",
    "previous_customer_response",
    "available_actions",
    "problem_type",
})

# Policy rule identifiers (machine-readable, stored in GuardrailResult.policy_rule).
RULE_ACTION_ALLOWED: str = "ACTION_ALLOWED"
RULE_INVALID_ACTION_ID: str = "INVALID_ACTION_ID"
RULE_PROHIBITED_ACTION: str = "PROHIBITED_ACTION"
RULE_MALFORMED_OBSERVATION: str = "MALFORMED_OBSERVATION"


# ---------------------------------------------------------------------------
# GuardrailResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardrailResult:
    """Structured result of a single guardrail decision.

    Attributes:
        original_action: The action that was proposed for evaluation.
            ``None`` if the action was not a valid integer (e.g. a string
            or float).  If the action was an integer but out of range, the
            integer value is still recorded so callers can see what was
            attempted.
        final_action: The action to execute — the original action if it was
            allowed, or the fallback action (ESCALATE by default) if it was
            blocked.
        allowed: ``True`` if the original action passed all checks and is
            safe to execute; ``False`` if it was blocked.
        reason: Human-readable explanation of the decision.
        policy_rule: Machine-readable policy rule identifier (one of the
            ``RULE_*`` constants defined in this module).
    """

    original_action: int | None
    final_action: int
    allowed: bool
    reason: str
    policy_rule: str


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------


def _coerce_action(action: Any) -> int | None:
    """Attempt to coerce *action* to a plain ``int``.

    Accepts Python ``int`` and NumPy integer types.  Rejects ``bool``
    (which is a subclass of ``int`` but semantically not an action),
    floats, strings, ``None``, and any other type.

    Returns:
        The action as a plain ``int``, or ``None`` if it cannot be coerced.
    """
    # bool is a subclass of int — reject it explicitly.
    if isinstance(action, bool):
        return None
    if isinstance(action, (int, np.integer)):
        return int(action)
    return None


def _validate_observation(observation: Any) -> tuple[bool, str]:
    """Validate that *observation* is well-formed.

    Checks (in order):
        1. ``observation`` is not ``None``.
        2. ``observation`` is a ``dict``.
        3. All eight required keys are present.
        4. ``available_actions`` is array-like with length ``NUM_ACTIONS``.
        5. ``problem_type`` is a valid scenario ID (0 to ``NUM_SCENARIOS-1``).

    Only the fields the guardrail actually needs are type-checked.
    Other fields (frustration, turn_count, etc.) are not validated because
    the guardrail does not use them for its decision.

    Returns:
        ``(is_valid, error_reason)``.  If valid, *error_reason* is ``""``.
    """
    if observation is None:
        return False, "observation is None"

    if not isinstance(observation, dict):
        return (
            False,
            f"observation is not a dict (got {type(observation).__name__})",
        )

    # All required keys present?
    missing = _REQUIRED_OBSERVATION_KEYS - set(observation.keys())
    if missing:
        return False, f"observation missing required keys: {sorted(missing)}"

    # available_actions: array-like, length NUM_ACTIONS
    available = observation["available_actions"]
    if available is None:
        return False, "available_actions is None"
    try:
        available_arr = np.asarray(available)
    except Exception:
        return False, "available_actions cannot be converted to array"
    if available_arr.ndim != 1 or available_arr.shape[0] != NUM_ACTIONS:
        return (
            False,
            f"available_actions has shape {available_arr.shape}, "
            f"expected ({NUM_ACTIONS},)",
        )

    # problem_type: valid scenario ID
    problem_type = observation["problem_type"]
    if problem_type is None:
        return False, "problem_type is None"
    try:
        pt = int(problem_type)
    except (TypeError, ValueError):
        return (
            False,
            f"problem_type is not an integer "
            f"(got {type(problem_type).__name__})",
        )
    if not (0 <= pt < NUM_SCENARIOS):
        return (
            False,
            f"problem_type {pt} is not a valid scenario ID "
            f"(must be 0-{NUM_SCENARIOS - 1})",
        )

    return True, ""


def _is_action_available(action: int, observation: dict) -> bool:
    """Check whether *action* is available for the current scenario.

    Uses two independent sources (defense in depth):
        1. Scenario definition: ``SCENARIOS[problem_type].available_actions``
        2. Observation mask: ``observation["available_actions"][action] == 1``

    The action is available **only if both sources agree**.  If either
    source says the action is not available, it is treated as prohibited.
    This catches cases where the observation mask has been corrupted or
    is inconsistent with the scenario definition.

    Args:
        action: The validated action integer (already in [0, NUM_ACTIONS)).
        observation: The validated observation dict.

    Returns:
        ``True`` if the action is available, ``False`` if prohibited.
    """
    problem_type = int(observation["problem_type"])
    scenario = SCENARIOS[problem_type]

    # Source 1: scenario definition (authoritative).
    in_scenario = action in scenario.available_actions

    # Source 2: observation mask (derived, checked for consistency).
    available_arr = np.asarray(observation["available_actions"])
    in_mask = int(available_arr[action]) == 1

    return in_scenario and in_mask


# ---------------------------------------------------------------------------
# Public API: GuardrailPolicy
# ---------------------------------------------------------------------------


class GuardrailPolicy:
    """Deterministic guardrail policy for evaluating agent actions.

    Validates actions against the observation and scenario definitions before
    they reach the environment.  When an action is blocked, the policy returns
    ESCALATE as the safe fallback action.

    ESCALATE (action 5) is available in all five scenarios, making it a safe
    universal fallback.  Escalating to a human agent is the appropriate response
    when the guardrail cannot validate the proposed action.

    This policy is fully deterministic: the same inputs always produce the same
    decision.  No LLM, external API, network request, or randomness is used.

    Usage::

        policy = GuardrailPolicy()
        result = policy.evaluate(action, observation)
        if result.allowed:
            env.step(result.final_action)
        else:
            logger.warning(result.reason)
            env.step(result.final_action)  # safe fallback
    """

    def __init__(self, fallback_action: int = ESCALATE) -> None:
        """Initialise the guardrail policy.

        Args:
            fallback_action: The action to use when an action is blocked.
                Defaults to ``ESCALATE``, which is available in all scenarios.

        Raises:
            ValueError: If *fallback_action* is not a valid action integer
                in ``[0, NUM_ACTIONS)``.
        """
        if isinstance(fallback_action, bool) or not isinstance(
            fallback_action, (int, np.integer)
        ):
            raise ValueError(
                f"fallback_action must be an integer, "
                f"got {type(fallback_action).__name__}"
            )
        if not (0 <= int(fallback_action) < NUM_ACTIONS):
            raise ValueError(
                f"fallback_action must be in [0, {NUM_ACTIONS}), "
                f"got {fallback_action}"
            )
        self._fallback_action: int = int(fallback_action)

    @property
    def fallback_action(self) -> int:
        """The action used when an action is blocked."""
        return self._fallback_action

    def evaluate(self, action: Any, observation: Any) -> GuardrailResult:
        """Evaluate *action* against the guardrail policy.

        This is the main entry point.  It validates the observation, checks
        the action ID, verifies action availability, and returns a structured
        decision result.

        Args:
            action: The proposed action (should be an integer in
                ``[0, NUM_ACTIONS)``).
            observation: The current observation dict from the environment.

        Returns:
            A :class:`GuardrailResult` describing the decision.
        """
        # --- Step 1: Validate observation ---
        obs_valid, obs_reason = _validate_observation(observation)
        if not obs_valid:
            original = _coerce_action(action)
            return GuardrailResult(
                original_action=original,
                final_action=self._fallback_action,
                allowed=False,
                reason=(
                    f"Action blocked due to malformed observation: "
                    f"{obs_reason}. "
                    f"Falling back to "
                    f"{ACTION_NAMES.get(self._fallback_action, 'UNKNOWN')}."
                ),
                policy_rule=RULE_MALFORMED_OBSERVATION,
            )

        # --- Step 2: Validate action ID ---
        action_int = _coerce_action(action)
        if action_int is None:
            return GuardrailResult(
                original_action=None,
                final_action=self._fallback_action,
                allowed=False,
                reason=(
                    f"Action {action!r} is not an integer. "
                    f"Must be an integer in [0, {NUM_ACTIONS}). "
                    f"Falling back to "
                    f"{ACTION_NAMES.get(self._fallback_action, 'UNKNOWN')}."
                ),
                policy_rule=RULE_INVALID_ACTION_ID,
            )
        if not (0 <= action_int < NUM_ACTIONS):
            return GuardrailResult(
                original_action=action_int,
                final_action=self._fallback_action,
                allowed=False,
                reason=(
                    f"Action {action_int} is out of range. "
                    f"Must be an integer in [0, {NUM_ACTIONS}). "
                    f"Falling back to "
                    f"{ACTION_NAMES.get(self._fallback_action, 'UNKNOWN')}."
                ),
                policy_rule=RULE_INVALID_ACTION_ID,
            )

        # --- Step 3: Check action availability ---
        if not _is_action_available(action_int, observation):
            action_name = ACTION_NAMES.get(action_int, f"UNKNOWN({action_int})")
            problem_type = int(observation["problem_type"])
            scenario_name = SCENARIOS[problem_type].name
            return GuardrailResult(
                original_action=action_int,
                final_action=self._fallback_action,
                allowed=False,
                reason=(
                    f"Action {action_name} ({action_int}) is not available "
                    f"for scenario '{scenario_name}' (id={problem_type}). "
                    f"Falling back to "
                    f"{ACTION_NAMES.get(self._fallback_action, 'UNKNOWN')}."
                ),
                policy_rule=RULE_PROHIBITED_ACTION,
            )

        # --- Step 4: Action is allowed ---
        action_name = ACTION_NAMES.get(action_int, f"UNKNOWN({action_int})")
        return GuardrailResult(
            original_action=action_int,
            final_action=action_int,
            allowed=True,
            reason=(
                f"Action {action_name} ({action_int}) is allowed for the "
                f"current scenario."
            ),
            policy_rule=RULE_ACTION_ALLOWED,
        )

    def guard(self, action: Any, observation: Any) -> int:
        """Evaluate *action* and return only the final action to execute.

        Convenience wrapper around :meth:`evaluate` for cases where only the
        action integer is needed (not the full decision result).

        Args:
            action: The proposed action.
            observation: The current observation dict.

        Returns:
            The action integer to execute (original if allowed, fallback if
            blocked).
        """
        return self.evaluate(action, observation).final_action


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

# Default policy instance used by the module-level convenience functions.
# ESCALATE is the fallback because it is available in all five scenarios.
_default_policy = GuardrailPolicy()


def evaluate_action(action: Any, observation: Any) -> GuardrailResult:
    """Evaluate *action* using the default guardrail policy.

    Args:
        action: The proposed action.
        observation: The current observation dict.

    Returns:
        A :class:`GuardrailResult` describing the decision.
    """
    return _default_policy.evaluate(action, observation)


def guard(action: Any, observation: Any) -> int:
    """Evaluate *action* and return the final action to execute.

    Args:
        action: The proposed action.
        observation: The current observation dict.

    Returns:
        The action integer to execute (original if allowed, ESCALATE if
        blocked).
    """
    return _default_policy.guard(action, observation)
