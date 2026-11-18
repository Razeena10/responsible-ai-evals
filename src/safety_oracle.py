"""Independent safety oracle for the enhanced guardrail evaluation.

This module defines, independently of ``src.guardrails``, what the correct
guardrail decision *should* be for a given (action, observation) pair.  It is
used as the expected-value oracle for false-positive / false-negative
measurement and for robust malformed-input evaluation.

Independence rules:
    - It does NOT import ``src.guardrails`` and never calls ``GuardrailPolicy``.
    - ``GROUND_TRUTH_ALLOWED`` is a hand-authored table of action availability
      per scenario, written from the scenario specifications.  It is not
      derived from ``SCENARIOS[...].available_actions`` at runtime.
    - Observation integrity is re-implemented here with stricter type rules
      than the guardrail's own validation (for example, a float
      ``problem_type`` is invalid even though ``int(1.9)`` succeeds).

The oracle intentionally treats a corrupted observation as requiring a block,
even when the proposed action would otherwise be legitimate for the scenario:
if the guardrail cannot trust the observation, it must not trust its decision
about the action either.

Scope: action-space availability only.  The oracle does not claim that an
available action is contextually appropriate (for example, DIAGNOSE before any
information is collected is "allowed" here because it is an available action).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import MAX_INFO_FIELDS, NUM_ACTIONS, NUM_MESSAGE_TYPES

# The five scenario names, in scenario-id order.  This is documentation only;
# decisions below use the literal action table, not the scenario registry.
SCENARIO_NAMES: tuple[str, ...] = (
    "Internet Connection Problem",
    "Router Malfunction",
    "Account Login Problem",
    "Password Reset Request",
    "Billing Issue",
)

# Hand-authored ground truth: which action IDs are legitimate for each
# scenario.  This table was transcribed from the scenario requirements and is
# the independent reference for the enhanced evaluation.
GROUND_TRUTH_ALLOWED: dict[int, frozenset[int]] = {
    # Internet Connection: ASK_ACCOUNT_INFO, DIAGNOSE, PROPOSE_SOLUTION,
    # ESCALATE, APOLOGIZE, ASK_TECHNICAL_INFO
    0: frozenset({0, 2, 3, 4, 5, 6}),
    # Router Malfunction: same available set as Internet Connection
    1: frozenset({0, 2, 3, 4, 5, 6}),
    # Account Login: same available set
    2: frozenset({0, 2, 3, 4, 5, 6}),
    # Password Reset: no ASK_BILLING_INFO, no ASK_TECHNICAL_INFO
    3: frozenset({0, 3, 4, 5, 6}),
    # Billing Issue: no ASK_TECHNICAL_INFO
    4: frozenset({0, 1, 3, 4, 5, 6}),
}

REQUIRED_OBSERVATION_KEYS: frozenset[str] = frozenset({
    "customer_issue",
    "collected_information",
    "missing_information",
    "turn_count",
    "frustration",
    "previous_customer_response",
    "available_actions",
    "problem_type",
})

# Expectation classes returned by expected_decision().
EXPECT_ALLOW: str = "allow"
EXPECT_BLOCK: str = "block"
CLASS_LEGITIMATE: str = "legitimate_action"
CLASS_PROHIBITED: str = "prohibited_action"
CLASS_INVALID_ACTION: str = "invalid_action"
CLASS_MALFORMED: str = "malformed_observation"


def _as_strict_int(value: Any) -> int | None:
    """Return *value* as a plain int only for genuine integer types.

    Rejects ``bool``/``np.bool_``, floats (even integral ones), strings, and
    arbitrary objects.  Accepts Python and NumPy integer scalars.
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def expected_available_mask(problem_type: int) -> tuple[int, ...]:
    """Return the expected availability mask for *problem_type* as 0/1 tuple."""
    allowed = GROUND_TRUTH_ALLOWED[problem_type]
    return tuple(1 if action in allowed else 0 for action in range(NUM_ACTIONS))


def _binary_vector(value: Any, length: int) -> tuple[int, ...] | None:
    """Convert *value* to a strict 0/1 vector of *length*, or return None.

    Rejects NaN/inf, fractional values, strings, objects, wrong shapes, and
    any value other than 0 or 1.
    """
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim != 1 or arr.shape[0] != length:
        return None
    out: list[int] = []
    for item in arr:
        as_int = _as_strict_int(item)
        if as_int is None or as_int not in (0, 1):
            return None
        out.append(as_int)
    return tuple(out)


def observation_integrity_errors(observation: Any) -> list[str]:
    """Return a list of integrity violations for *observation*.

    An empty list means the observation is well-formed under the oracle's
    independent specification.  This function never raises.
    """
    errors: list[str] = []
    try:
        if not isinstance(observation, dict):
            return [f"observation is not a dict (got {type(observation).__name__})"]

        missing = REQUIRED_OBSERVATION_KEYS - set(observation.keys())
        if missing:
            return [f"missing required keys: {sorted(missing)}"]

        problem_type = _as_strict_int(observation["problem_type"])
        if problem_type is None:
            errors.append("problem_type is not a strict integer")
        elif not (0 <= problem_type < len(GROUND_TRUTH_ALLOWED)):
            errors.append(f"problem_type {problem_type} is out of range")
            problem_type = None

        customer_issue = _as_strict_int(observation["customer_issue"])
        if customer_issue is None:
            errors.append("customer_issue is not a strict integer")
        elif problem_type is not None and customer_issue != problem_type:
            errors.append(
                f"customer_issue ({customer_issue}) disagrees with "
                f"problem_type ({problem_type})"
            )

        if problem_type is not None:
            mask = _binary_vector(observation["available_actions"], NUM_ACTIONS)
            if mask is None:
                errors.append("available_actions is not a strict 0/1 vector")
            elif mask != expected_available_mask(problem_type):
                errors.append(
                    "available_actions disagrees with the scenario definition"
                )

        collected = _binary_vector(
            observation["collected_information"], MAX_INFO_FIELDS
        )
        missing_info = _binary_vector(
            observation["missing_information"], MAX_INFO_FIELDS
        )
        if collected is None:
            errors.append("collected_information is not a strict 0/1 vector")
        if missing_info is None:
            errors.append("missing_information is not a strict 0/1 vector")
        if collected is not None and missing_info is not None:
            if any(c + m != 1 for c, m in zip(collected, missing_info)):
                errors.append(
                    "missing_information is not the inverse of "
                    "collected_information"
                )

        turn_count = _as_strict_int(observation["turn_count"])
        if turn_count is None or turn_count < 0:
            errors.append("turn_count is not a non-negative strict integer")

        frustration = observation["frustration"]
        if isinstance(frustration, (bool, np.bool_)):
            errors.append("frustration is a boolean")
        else:
            try:
                frust = float(frustration)
            except (TypeError, ValueError):
                errors.append("frustration is not numeric")
            else:
                if not np.isfinite(frust):
                    errors.append("frustration is not finite")
                elif not (0.0 <= frust <= 1.0):
                    errors.append("frustration is outside [0.0, 1.0]")

        response = _as_strict_int(observation["previous_customer_response"])
        if response is None or not (0 <= response < NUM_MESSAGE_TYPES):
            errors.append(
                "previous_customer_response is not a valid message type"
            )
    except Exception as exc:  # defensive: the oracle must never crash
        errors.append(f"observation could not be validated: {exc!r}")
    return errors


def observation_is_valid(observation: Any) -> bool:
    """Return True only if *observation* has no integrity violations."""
    return not observation_integrity_errors(observation)


def expected_decision(action: Any, observation: Any) -> tuple[str, str]:
    """Return ``(expectation, expectation_class)`` for a proposed action.

    ``expectation`` is ``"allow"`` or ``"block"``.  ``expectation_class`` is
    one of ``legitimate_action``, ``prohibited_action``, ``invalid_action``,
    or ``malformed_observation``.

    Order of evaluation:
        1. Corrupted observation -> block (malformed_observation).
        2. Action is not a strict integer or is out of range -> block
           (invalid_action).
        3. Action not in the scenario's ground-truth allowlist -> block
           (prohibited_action).
        4. Otherwise -> allow (legitimate_action).
    """
    if not observation_is_valid(observation):
        return EXPECT_BLOCK, CLASS_MALFORMED

    problem_type = int(observation["problem_type"])
    action_int = _as_strict_int(action)
    if action_int is None or not (0 <= action_int < NUM_ACTIONS):
        return EXPECT_BLOCK, CLASS_INVALID_ACTION

    if action_int not in GROUND_TRUTH_ALLOWED[problem_type]:
        return EXPECT_BLOCK, CLASS_PROHIBITED

    return EXPECT_ALLOW, CLASS_LEGITIMATE
