"""Guardrail wrapper for end-to-end evaluation.

This module places :class:`~src.guardrails.GuardrailPolicy` between an agent and
the environment so that **every** proposed action is evaluated before it reaches
``CustomerSupportEnv.step``.  It does not modify the Week 3 agent or environment;
it composes them.

The wrapper is also the measuring instrument for the end-to-end evaluation:

    - it records every proposed action and the guardrail decision,
    - it records the action that was actually executed,
    - it proves (per step) that a blocked original action was never executed,
    - it counts fallback escalations, unnecessary escalations (over-blocks of
      legitimate actions), and invalid actions that reached the environment.

Definitions used here (fixed before measurement, per project requirements):

    blocked action              guardrail returned ``allowed=False``
    fallback escalation         blocked action whose fallback action is ESCALATE
    unnecessary escalation      a blocked action that an independent oracle says
                                was a legitimate action (i.e. an over-block that
                                caused an escalation)
    invalid action reaching env executed action is outside ``env.action_space``
    prevention violation        a blocked original action was executed anyway

The optional ``observation_mutator`` exists only for diagnostics: it corrupts
the observation *seen by the guardrail* while leaving the environment's own
observation untouched.  With it, an evaluation can check that the guardrail
fails safe (blocks) when the observation it is given is untrustworthy.  The
default (``None``) passes the environment observation through unchanged.

Scope notes:
    - The agent is deterministic and rule-based; this is not an LLM guardrail.
    - The policy is stateless: it judges one action at a time.
    - The default fallback (ESCALATE) terminates the episode, so repeated
      blocked actions across turns are only observable in a live loop when a
      non-terminating fallback is configured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from src.config import ESCALATE, ACTION_NAMES, NUM_ACTIONS
from src.guardrails import GuardrailPolicy, GuardrailResult
from src.safety_oracle import (
    CLASS_LEGITIMATE,
    GROUND_TRUTH_ALLOWED,
    expected_decision,
)
from src.support_env import CustomerSupportEnv


def action_token(action: Any) -> str:
    """Return a stable, human-readable representation of a proposed action.

    Used instead of ``repr`` so that generated cases containing arbitrary
    objects (whose ``repr`` includes a memory address) stay reproducible.
    """
    if isinstance(action, np.generic):
        return f"{type(action).__name__}({action.item()!r})"
    if isinstance(action, (bool, int, float, str, bytes, type(None))):
        return f"{type(action).__name__}:{action!r}"
    if isinstance(action, (list, tuple)):
        return f"{type(action).__name__}:{action!r}"
    return f"{type(action).__name__}"


# ---------------------------------------------------------------------------
# StepRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    """One proposed action and everything that happened to it.

    Attributes:
        episode: Episode index within the enclosing evaluation run.
        step: Zero-based step index within the episode.
        proposed_action: Stable string representation of the proposed action.
        proposed_action_is_int: Whether the proposal was an integer.
        original_action: Guardrail's recorded original action (int or None).
        allowed: Whether the guardrail allowed the proposal.
        final_action: Action returned by the guardrail.
        final_action_name: Human-readable name of ``final_action``.
        executed_action: Action actually passed to ``env.step``.
        policy_rule: Guardrail policy rule identifier.
        expected: Independent-oracle expectation ("allow" / "block").
        expected_class: Independent-oracle expectation class.
        fallback_escalation: Blocked and the fallback was ESCALATE.
        unnecessary_escalation: Blocked although the action was legitimate.
        prevention_violation: Blocked original action was executed anyway.
        policy_error: Exception type name if the guardrail itself raised,
            else None.  A policy error is treated as a block and replaced by
            the configured fallback, so the environment still receives a
            valid action.
        env_error: Exception type name if ``env.step`` raised, else None.
    """

    episode: int
    step: int
    proposed_action: str
    proposed_action_is_int: bool
    original_action: int | None
    allowed: bool
    final_action: int
    final_action_name: str
    executed_action: int
    policy_rule: str
    expected: str
    expected_class: str
    fallback_escalation: bool
    unnecessary_escalation: bool
    prevention_violation: bool
    policy_error: str | None
    env_error: str | None


# ---------------------------------------------------------------------------
# Agents used to exercise the guarded path
# ---------------------------------------------------------------------------


class ProhibitedProbeAgent:
    """Baseline agent that injects one prohibited action on turn 2.

    The injected action is chosen from the independent oracle's ground-truth
    table (never from the guardrail), so the probe is genuinely prohibited in
    the current scenario.  On all other turns it delegates to
    :class:`~src.baseline_agent.BaselineSupportAgent`.

    In the Billing scenario every action is available, so no prohibited
    action exists there; the probe cannot inject and simply delegates to the
    baseline agent.  That absence is reported rather than hidden.
    """

    def __init__(self) -> None:
        from src.baseline_agent import BaselineSupportAgent

        self._baseline = BaselineSupportAgent()
        self._injected = False

    def reset(self) -> None:
        self._baseline.reset()
        self._injected = False

    def select_action(self, observation: dict) -> int:
        problem_type = int(observation["problem_type"])
        if (
            not self._injected
            and int(observation["turn_count"]) == 2
        ):
            self._injected = True
            prohibited = sorted(
                set(range(NUM_ACTIONS)) - set(GROUND_TRUTH_ALLOWED[problem_type])
            )
            if prohibited:
                return prohibited[0]
        return self._baseline.select_action(observation)


class AlwaysProhibitedAgent:
    """Deterministic agent that always proposes an action prohibited in context.

    The prohibited action is taken from the independent oracle table, so the
    agent never reads the guardrail.  It has no anti-repetition logic: the
    point is to exercise repeated blocked proposals.
    """

    def __init__(self) -> None:
        self._action: int = 0

    def reset(self) -> None:
        self._action = 0

    def select_action(self, observation: dict) -> int:
        problem_type = int(observation["problem_type"])
        prohibited = sorted(
            set(range(NUM_ACTIONS)) - set(GROUND_TRUTH_ALLOWED[problem_type])
        )
        # Every scenario has at least one prohibited action by construction.
        self._action = prohibited[0]
        return self._action


# ---------------------------------------------------------------------------
# GuardrailWrapper
# ---------------------------------------------------------------------------


class GuardrailWrapper:
    """Wrap an environment so every action passes through a guardrail policy.

    Args:
        env: The environment to wrap.  It is not modified.
        policy: Guardrail policy to apply.  Defaults to ``GuardrailPolicy()``
            (ESCALATE fallback).
        observation_mutator: Optional callable applied to the observation
            *before* the guardrail sees it.  Diagnostic use only; the
            environment always receives its own unmodified observation.
        use_oracle: Whether to consult the independent oracle for each step.
            When True, ``unnecessary_escalation`` is measured against the
            oracle rather than against the guardrail's own decision.

    Attributes:
        step_records: All :class:`StepRecord` entries from this wrapper.
    """

    def __init__(
        self,
        env: CustomerSupportEnv,
        policy: GuardrailPolicy | None = None,
        observation_mutator: Callable[[dict], Any] | None = None,
        use_oracle: bool = True,
    ) -> None:
        self._env = env
        self._policy = policy or GuardrailPolicy()
        self._observation_mutator = observation_mutator
        self._use_oracle = use_oracle

        self._episode = 0
        self._step = 0
        self._obs: dict | None = None
        self._episode_open = False

        self.step_records: list[StepRecord] = []
        self.blocked_actions = 0
        self.fallback_escalations = 0
        self.unnecessary_escalations = 0
        self.invalid_actions_reaching_env = 0
        self.prevention_violations = 0
        self.policy_errors = 0
        self.env_errors = 0
        self.episodes_terminated_due_to_guardrail = 0
        self.episodes = 0
        self.steps = 0

    # --- Environment-like API --------------------------------------------

    @property
    def action_space(self):
        """Delegate the environment's action space."""
        return self._env.action_space

    @property
    def observation_space(self):
        """Delegate the environment's observation space."""
        return self._env.observation_space

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        """Start a new episode; the wrapper is not re-created between episodes."""
        observation, info = self._env.reset(seed=seed, options=options)
        self._obs = observation
        self._step = 0
        self._episode_open = True
        self.episodes += 1
        self._episode = self.episodes - 1
        return observation, info

    def step(
        self, action: Any
    ) -> tuple[dict, float, bool, bool, dict]:
        """Evaluate *action*, execute the guarded action, and record the step.

        Raises:
            RuntimeError: If called before :meth:`reset`.
            Exception: Any exception raised by the environment is re-raised
                after being recorded.
        """
        if not self._episode_open or self._obs is None:
            raise RuntimeError("Call reset() before step().")

        guardrail_observation = self._obs
        if self._observation_mutator is not None:
            guardrail_observation = self._observation_mutator(self._obs)

        policy_error: str | None = None
        try:
            result = self._policy.evaluate(action, guardrail_observation)
        except Exception as exc:
            # Fail safe at the harness boundary: a guardrail crash is treated
            # as a block and replaced by the configured fallback action.
            policy_error = type(exc).__name__
            result = GuardrailResult(
                original_action=(
                    int(action)
                    if isinstance(action, (int, np.integer))
                    and not isinstance(action, bool)
                    else None
                ),
                final_action=self._policy.fallback_action,
                allowed=False,
                reason=f"Guardrail raised {policy_error}; using fallback.",
                policy_rule=f"EXCEPTION:{policy_error}",
            )
        executed = result.final_action

        if self._use_oracle:
            expected, expected_class = expected_decision(
                action, guardrail_observation
            )
        else:
            expected, expected_class = "unknown", "unknown"

        blocked = not result.allowed
        fallback_escalation = bool(blocked and executed == ESCALATE)
        unnecessary_escalation = bool(
            blocked and expected_class == CLASS_LEGITIMATE
        )
        prevention_violation = bool(
            blocked
            and isinstance(result.original_action, int)
            and executed == result.original_action
            and executed != result.final_action
        )

        env_error: str | None = None
        try:
            observation, reward, terminated, truncated, info = self._env.step(
                executed
            )
        except Exception as exc:  # recorded, then re-raised unchanged
            env_error = type(exc).__name__
            self.env_errors += 1
            self._record(
                action=action,
                result=result,
                executed=executed,
                expected=expected,
                expected_class=expected_class,
                fallback_escalation=fallback_escalation,
                unnecessary_escalation=unnecessary_escalation,
                prevention_violation=prevention_violation,
                policy_error=policy_error,
                env_error=env_error,
            )
            self._episode_open = False
            raise

        # An action outside the action space is an invalid action that reached
        # the environment.  (The shipped policy prevents this; the counter
        # exists so a defective policy cannot pass silently.)
        if not isinstance(executed, (int, np.integer)) or not (
            0 <= int(executed) < NUM_ACTIONS
        ):
            self.invalid_actions_reaching_env += 1

        if policy_error is not None:
            self.policy_errors += 1
        if blocked:
            self.blocked_actions += 1
        if fallback_escalation:
            self.fallback_escalations += 1
        if unnecessary_escalation:
            self.unnecessary_escalations += 1
        if prevention_violation:
            self.prevention_violations += 1
        if (terminated or truncated) and blocked:
            self.episodes_terminated_due_to_guardrail += 1
        if terminated or truncated:
            self._episode_open = False

        self._obs = observation
        self.steps += 1
        self._record(
            action=action,
            result=result,
            executed=executed,
            expected=expected,
            expected_class=expected_class,
            fallback_escalation=fallback_escalation,
            unnecessary_escalation=unnecessary_escalation,
            prevention_violation=prevention_violation,
            policy_error=policy_error,
            env_error=None,
        )
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Close the wrapped environment."""
        self._env.close()

    # --- Reporting --------------------------------------------------------

    def summary(self) -> dict:
        """Return aggregate counters for this wrapper's run."""
        return {
            "episodes": self.episodes,
            "steps": self.steps,
            "blocked_actions": self.blocked_actions,
            "fallback_escalations": self.fallback_escalations,
            "unnecessary_escalations": self.unnecessary_escalations,
            "invalid_actions_reaching_env": self.invalid_actions_reaching_env,
            "prevention_violations": self.prevention_violations,
            "policy_errors": self.policy_errors,
            "env_errors": self.env_errors,
            "episodes_terminated_due_to_guardrail": (
                self.episodes_terminated_due_to_guardrail
            ),
        }

    def step_records_as_dicts(self) -> list[dict]:
        """Return all step records as plain dicts."""
        return [asdict(r) for r in self.step_records]

    # --- Internal ---------------------------------------------------------

    def _record(
        self,
        *,
        action: Any,
        result: Any,
        executed: int,
        expected: str,
        expected_class: str,
        fallback_escalation: bool,
        unnecessary_escalation: bool,
        prevention_violation: bool,
        policy_error: str | None,
        env_error: str | None,
    ) -> None:
        self.step_records.append(
            StepRecord(
                episode=self._episode,
                step=self._step,
                proposed_action=action_token(action),
                proposed_action_is_int=isinstance(
                    action, (int, np.integer)
                ) and not isinstance(action, bool),
                original_action=result.original_action,
                allowed=result.allowed,
                final_action=result.final_action,
                final_action_name=ACTION_NAMES.get(
                    result.final_action, f"UNKNOWN({result.final_action})"
                ),
                executed_action=executed,
                policy_rule=result.policy_rule,
                expected=expected,
                expected_class=expected_class,
                fallback_escalation=fallback_escalation,
                unnecessary_escalation=unnecessary_escalation,
                prevention_violation=prevention_violation,
                policy_error=policy_error,
                env_error=env_error,
            )
        )
        self._step += 1


def make_oracle_mutator(
    corruption: str,
) -> Callable[[dict], Any]:
    """Build a diagnostic observation mutator for the guardrail input.

    The environment always keeps its own valid observation; only the copy
    handed to the guardrail is corrupted.  This tests fail-safe behaviour.

    Args:
        corruption: One of ``"clear_mask"``, ``"wrong_problem_type"``,
            ``"collection_length"``, ``"nan_frustration"``.

    Returns:
        A callable that maps an observation dict to a corrupted copy.

    Raises:
        ValueError: If *corruption* is unknown.
    """
    def mutate(observation: dict) -> Any:
        corrupted = dict(observation)
        if corruption == "clear_mask":
            corrupted["available_actions"] = np.zeros(
                NUM_ACTIONS, dtype=np.int8
            )
        elif corruption == "wrong_problem_type":
            corrupted["problem_type"] = np.int64(99)
        elif corruption == "collection_length":
            corrupted["collected_information"] = np.asarray(
                observation["collected_information"]
            )[:2]
        elif corruption == "nan_frustration":
            corrupted["frustration"] = np.array(np.nan, dtype=np.float32)
        else:  # pragma: no cover - guarded by the validation below
            raise ValueError(f"unknown corruption: {corruption}")
        return corrupted

    return mutate


__all__ = [
    "AlwaysProhibitedAgent",
    "GuardrailWrapper",
    "ProhibitedProbeAgent",
    "StepRecord",
    "action_token",
    "make_oracle_mutator",
]
