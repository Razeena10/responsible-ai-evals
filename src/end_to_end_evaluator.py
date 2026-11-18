"""End-to-end guardrail evaluation (Phase 4).

Every configuration runs the same deterministic agent against the Week 3
environment.  Guarded configurations place :class:`GuardrailWrapper` between
the agent and ``CustomerSupportEnv.step`` so **every** proposed action is
evaluated by :class:`~src.guardrails.GuardrailPolicy` before the environment
sees it.  The agent and the environment are imported, never modified.

Configurations
--------------
1. ``baseline_unguarded``
       BaselineSupportAgent, no guardrail.  The reference point.
2. ``baseline_guarded``
       Same agent and seeds, every action evaluated by the default policy
       (ESCALATE fallback).
3. ``prohibited_probe_unguarded``
       ProhibitedProbeAgent injects one oracle-prohibited action on turn 2.
       Unguarded, to show what reaches the environment without the guardrail.
4. ``prohibited_probe_guarded``
       Same agent and seeds with the wrapper.  Blocked proposals must never be
       executed (``prevention_violations == 0``).  In the Billing scenario the
       injected action is actually legitimate, so a block there is counted as
       an unnecessary escalation rather than a success.
5. ``baseline_corrupted_observation``
       Diagnostic: the environment observation is valid, but the copy given to
       the guardrail has an out-of-range ``problem_type``.  Measures fail-safe
       behaviour on untrustworthy input.  This is a diagnostic harness
       configuration, not a shipped deployment mode.

``invalid_actions_reaching_env`` is counted by the wrapper as any executed
action outside ``env.action_space``; the agent is deterministic and only
proposes valid integers, so it is expected to be zero in all configurations.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from src.baseline_agent import BaselineSupportAgent
from src.config import ESCALATE, NUM_ACTIONS
from src.guardrails import GuardrailPolicy
from src.guardrail_wrapper import (
    GuardrailWrapper,
    ProhibitedProbeAgent,
    make_oracle_mutator,
)
from src.safety_oracle import CLASS_PROHIBITED, expected_decision
from src.statistics_utils import wilson_interval
from src.support_env import CustomerSupportEnv

_REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "reports"
)

EPISODES_PER_CONFIG = 100
BASE_SEED = 20260101
MAX_TURNS = 8

CONFIG_NAMES = (
    "baseline_unguarded",
    "baseline_guarded",
    "prohibited_probe_unguarded",
    "prohibited_probe_guarded",
    "baseline_corrupted_observation",
)


def _as_int(action: Any) -> int | None:
    """Return *action* as a plain int, or None when it is not integral."""
    if isinstance(action, bool):
        return None
    if isinstance(action, np.integer):
        return int(action)
    if isinstance(action, int):
        return action
    return None


def _episode_seeds(count: int = EPISODES_PER_CONFIG) -> list[int]:
    return [BASE_SEED + i for i in range(count)]


def _empty_episode() -> dict:
    return {
        "success": False,
        "reward": 0.0,
        "turns": 0,
        "termination_reason": None,
        "scenario_id": None,
        "blocked": 0,
        "fallback_escalations": 0,
        "unnecessary_escalations": 0,
        "prohibited_actions_attempted": 0,
        "prohibited_actions_reaching_env": 0,
        "episode_terminated_by_guardrail": False,
    }


def run_unguarded_episode(
    agent: Any, seed: int, max_turns: int = MAX_TURNS
) -> dict:
    """Run one episode without a guardrail and return episode metrics."""
    env = CustomerSupportEnv(max_turns=max_turns)
    observation, reset_info = env.reset(seed=seed)
    agent.reset()
    result = _empty_episode()
    result["scenario_id"] = reset_info["scenario_id"]
    env_terminated = False

    try:
        while not env_terminated:
            proposed = agent.select_action(observation)
            action = _as_int(proposed)
            expected, expected_class = expected_decision(
                action, observation
            )
            if expected_class == CLASS_PROHIBITED:
                result["prohibited_actions_attempted"] += 1
                result["prohibited_actions_reaching_env"] += 1

            observation, reward, terminated, truncated, info = env.step(
                action
            )
            result["turns"] += 1
            result["reward"] += float(reward)
            if info.get("success"):
                result["success"] = True
            if terminated or truncated:
                result["termination_reason"] = info.get(
                    "termination_reason"
                )
                env_terminated = True
    finally:
        env.close()
    return result


def run_guarded_episode(
    agent: Any,
    seed: int,
    policy: GuardrailPolicy | None = None,
    observation_mutator: Any | None = None,
    max_turns: int = MAX_TURNS,
) -> dict:
    """Run one episode through the guardrail wrapper and return its metrics."""
    env = CustomerSupportEnv(max_turns=max_turns)
    wrapper = GuardrailWrapper(
        env,
        policy=policy or GuardrailPolicy(),
        observation_mutator=observation_mutator,
        use_oracle=True,
    )
    observation, reset_info = wrapper.reset(seed=seed)
    agent.reset()
    result = _empty_episode()
    result["scenario_id"] = reset_info["scenario_id"]
    env_terminated = False

    try:
        while not env_terminated:
            proposed = agent.select_action(observation)
            before = wrapper.summary()
            observation, reward, terminated, truncated, info = wrapper.step(
                proposed
            )
            after = wrapper.summary()
            step_record = wrapper.step_records[-1]

            result["turns"] += 1
            result["reward"] += float(reward)
            result["blocked"] += (
                after["blocked_actions"] - before["blocked_actions"]
            )
            result["fallback_escalations"] += (
                after["fallback_escalations"]
                - before["fallback_escalations"]
            )
            result["unnecessary_escalations"] += (
                after["unnecessary_escalations"]
                - before["unnecessary_escalations"]
            )
            if step_record.expected_class == CLASS_PROHIBITED:
                result["prohibited_actions_attempted"] += 1
            if step_record.expected_class == CLASS_PROHIBITED and (
                step_record.executed_action
                == step_record.original_action
            ):
                # Only possible if the guardrail allowed it.
                result["prohibited_actions_reaching_env"] += 1
            if info.get("success"):
                result["success"] = True
            if terminated or truncated:
                result["termination_reason"] = info.get(
                    "termination_reason"
                )
                result["episode_terminated_by_guardrail"] = bool(
                    step_record.policy_rule == "POLICY_FALLBACK"
                    or after["episodes_terminated_due_to_guardrail"]
                    > before["episodes_terminated_due_to_guardrail"]
                )
                env_terminated = True

        totals = wrapper.summary()
        result["prevention_violations"] = totals[
            "prevention_violations"
        ]
        result["invalid_actions_reaching_env"] = totals[
            "invalid_actions_reaching_env"
        ]
        result["policy_errors"] = totals["policy_errors"]
        result["env_errors"] = totals["env_errors"]
    finally:
        wrapper.close()
    return result


def _aggregate(episodes: list[dict]) -> dict:
    """Aggregate per-episode metrics into rates, means, and Wilson CIs."""
    n = len(episodes)
    successes = sum(1 for e in episodes if e["success"])
    rate = successes / n if n else 0.0
    low, high = wilson_interval(successes, n)
    prohibited_attempted = sum(
        e["prohibited_actions_attempted"] for e in episodes
    )
    prohibited_reached = sum(
        e["prohibited_actions_reaching_env"] for e in episodes
    )
    return {
        "episodes": n,
        "successes": successes,
        "success_rate": rate,
        "success_rate_95ci": [low, high],
        "avg_reward": (
            sum(e["reward"] for e in episodes) / n if n else 0.0
        ),
        "avg_turns": (
            sum(e["turns"] for e in episodes) / n if n else 0.0
        ),
        "blocked_actions": sum(e["blocked"] for e in episodes),
        "fallback_escalations": sum(
            e["fallback_escalations"] for e in episodes
        ),
        "unnecessary_escalations": sum(
            e["unnecessary_escalations"] for e in episodes
        ),
        "prohibited_actions_attempted": prohibited_attempted,
        "prohibited_actions_reaching_env": prohibited_reached,
        "prohibited_action_prevention_rate": (
            1.0 - prohibited_reached / prohibited_attempted
            if prohibited_attempted
            else None
        ),
        "invalid_actions_reaching_env": sum(
            e.get("invalid_actions_reaching_env", 0) for e in episodes
        ),
        "prevention_violations": sum(
            e.get("prevention_violations", 0) for e in episodes
        ),
        "policy_errors": sum(e.get("policy_errors", 0) for e in episodes),
        "env_errors": sum(e.get("env_errors", 0) for e in episodes),
        "episodes_terminated_by_guardrail": sum(
            1 for e in episodes if e["episode_terminated_by_guardrail"]
        ),
        "termination_reasons": _count_termination_reasons(episodes),
    }


def _count_termination_reasons(episodes: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in episodes:
        reason = episode["termination_reason"] or "none"
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _run_config(
    name: str,
    agent: Any,
    guarded: bool,
    observation_mutator: Any | None = None,
) -> dict:
    seeds = _episode_seeds()
    episodes = []
    for seed in seeds:
        if guarded:
            episodes.append(
                run_guarded_episode(
                    agent,
                    seed,
                    observation_mutator=observation_mutator,
                )
            )
        else:
            episodes.append(run_unguarded_episode(agent, seed))
    metrics = _aggregate(episodes)

    scenario_success: dict[str, dict[str, int]] = {}
    for episode in episodes:
        scenario = str(episode["scenario_id"])
        entry = scenario_success.setdefault(
            scenario, {"episodes": 0, "successes": 0}
        )
        entry["episodes"] += 1
        entry["successes"] += int(episode["success"])
    metrics["per_scenario"] = dict(sorted(scenario_success.items()))
    return {"config": name, "metrics": metrics}


def run_end_to_end_evaluation() -> dict:
    """Run every configuration and return the report payload."""
    results = {
        "baseline_unguarded": _run_config(
            "baseline_unguarded", BaselineSupportAgent(), guarded=False
        ),
        "baseline_guarded": _run_config(
            "baseline_guarded", BaselineSupportAgent(), guarded=True
        ),
        "prohibited_probe_unguarded": _run_config(
            "prohibited_probe_unguarded",
            ProhibitedProbeAgent(),
            guarded=False,
        ),
        "prohibited_probe_guarded": _run_config(
            "prohibited_probe_guarded",
            ProhibitedProbeAgent(),
            guarded=True,
        ),
        "baseline_corrupted_observation": _run_config(
            "baseline_corrupted_observation",
            BaselineSupportAgent(),
            guarded=True,
            observation_mutator=make_oracle_mutator("wrong_problem_type"),
        ),
    }

    base = results["baseline_unguarded"]["metrics"]
    guarded = results["baseline_guarded"]["metrics"]
    probe_base = results["prohibited_probe_unguarded"]["metrics"]
    probe_guarded = results["prohibited_probe_guarded"]["metrics"]

    deltas = {
        "guarded_minus_unguarded": {
            "success_rate": guarded["success_rate"]
            - base["success_rate"],
            "avg_reward": guarded["avg_reward"] - base["avg_reward"],
            "avg_turns": guarded["avg_turns"] - base["avg_turns"],
        },
        "probe_guardrail_effect": {
            "blocked_actions": probe_guarded["blocked_actions"]
            - probe_base["blocked_actions"],
            "prohibited_actions_reaching_env": (
                probe_guarded["prohibited_actions_reaching_env"]
                - probe_base["prohibited_actions_reaching_env"]
            ),
            "success_rate": probe_guarded["success_rate"]
            - probe_base["success_rate"],
            "avg_reward": probe_guarded["avg_reward"]
            - probe_base["avg_reward"],
            "avg_turns": probe_guarded["avg_turns"]
            - probe_base["avg_turns"],
        },
    }

    integrity = {
        "blocked_actions_never_reach_env": all(
            results[name]["metrics"]["prevention_violations"] == 0
            for name in (
                "baseline_guarded",
                "prohibited_probe_guarded",
                "baseline_corrupted_observation",
            )
        ),
        "invalid_actions_reaching_env": {
            name: results[name]["metrics"][
                "invalid_actions_reaching_env"
            ]
            for name in CONFIG_NAMES
        },
        "probe_prohibited_attempts": probe_base[
            "prohibited_actions_attempted"
        ],
        "note": (
            "A blocked original action is never executed: the wrapper "
            "records the action it passes to env.step and compares it to the "
            "guardrail's original action on every step."
        ),
    }

    return {
        "metadata": {
            "episodes_per_config": EPISODES_PER_CONFIG,
            "configs": list(CONFIG_NAMES),
            "seeds": {
                "base": BASE_SEED,
                "values": _episode_seeds(),
            },
            "max_turns": MAX_TURNS,
            "agent": "BaselineSupportAgent (deterministic, rule-based)",
            "environment": "CustomerSupportEnv (unchanged Week 3 code)",
            "policy": "GuardrailPolicy() default, ESCALATE fallback",
            "scope_note": (
                "This is a deterministic rule-based agent and a stateless "
                "action guardrail, not an LLM or a full safety system. "
                "Success is the environment's own RESOLVED termination. "
                "The corrupted-observation configuration is a diagnostic of "
                "fail-safe behaviour, not a deployment mode."
            ),
        },
        "configs": results,
        "deltas": deltas,
        "integrity_checks": integrity,
    }


def write_report(
    report: dict | None = None, output_dir: str = _REPORTS_DIR
) -> str:
    """Write ``reports/end_to_end_guardrail_report.json`` and return its path."""
    payload = report if report is not None else run_end_to_end_evaluation()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "end_to_end_guardrail_report.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main() -> int:
    payload = run_end_to_end_evaluation()
    path = write_report(payload)
    print(f"wrote {path}")
    for name in CONFIG_NAMES:
        metrics = payload["configs"][name]["metrics"]
        print(
            f"{name}: success={metrics['successes']}/"
            f"{metrics['episodes']} "
            f"avg_reward={metrics['avg_reward']:.3f} "
            f"avg_turns={metrics['avg_turns']:.2f} "
            f"blocked={metrics['blocked_actions']} "
            f"prohibited_reached_env="
            f"{metrics['prohibited_actions_reaching_env']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
