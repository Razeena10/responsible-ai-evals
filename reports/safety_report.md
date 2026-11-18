# Safety Evaluation Report — responsible-ai-evals

Generated: 2026-11-18T13:39:15.534419+00:00

> **Scope note:** The agent under evaluation is a deterministic, rule-based customer support agent — not an LLM. This evaluation tests action-space safety, not natural-language prompt injection or LLM-specific safety concerns.

---

## Evaluation Methodology

The safety evaluation tests the guardrail policy and the baseline support agent under four categories of challenging scenarios. The evaluation is fully deterministic and reproducible (seed=42). No LLM, external API, or network request is used.

| Category | Description |
|---|---|
| UNSAFE_ACTION | Prohibited and invalid action IDs |
| REPEATED_FAILURE | Repeated/unproductive actions and repetition safeguards |
| UNSUPPORTED_INPUT | Malformed observations and invalid inputs |
| RULE_BYPASS | Synthetic adversarial observations with inconsistent masks |

## Results and Metrics

| Metric | Value |
|---|---|
| Total test cases | 41 |
| Passed cases | 41 |
| Failed cases | 0 |

| Block rate (prohibited actions) | 6/6 = 100.0% |
| Safe-action acceptance rate | 6/6 = 100.0% |
| Invalid action block rate | 7/7 = 100.0% |
| Unsupported inputs handled safely | 10/10 = 100.0% |
| Rule-bypass block rate | 6/6 = 100.0% |
| Repeated-action risks identified | 3/6 |
| Repeated-action cases passed | 6/6 = 100.0% |

## Failed Cases

No failed cases. All test cases passed.

> Passing all test cases confirms the guardrail works as designed for the tested scenarios. It does not prove the system is safe under all conditions.

## Observed Weaknesses

1. Guardrail fallback (ESCALATE) terminates the episode — the system cannot continue helping the customer after a guardrail intervention.
2. The guardrail evaluates each action independently — it does not track whether the agent has been repeatedly attempting the same blocked action across multiple turns.
3. Defense-in-depth depends on problem_type accuracy — if the observation's problem_type is corrupted, the scenario-based check may allow an action that should not be available for the actual scenario.
4. Repetition safeguards live in the environment, not the guardrail — the guardrail does not prevent unproductive repetition of valid, available actions. The environment's penalty and the agent's anti-repetition logic handle this separately.

## Known Limitations

1. This evaluation tests a deterministic, discrete-action support agent — not an LLM. It does not test natural-language prompt injection, jailbreaks, or LLM-specific safety concerns.
2. The evaluation covers predefined scenarios and test cases. It does not test stochastic scenarios, production edge cases, or novel attack vectors.
3. Passing all test cases confirms the guardrail works as designed for the tested cases. It does not prove the system is safe under all conditions.
4. The evaluation is deterministic and reproducible. It does not account for random failures, race conditions, or production environment issues.

## Recommended Mitigations

1. Add cross-turn action tracking to the guardrail to detect agents that repeatedly attempt blocked actions.
2. Consider a non-terminating fallback action (e.g., APOLOGIZE) for guardrail interventions that do not require immediate escalation.
3. Add integrity checks on the observation's problem_type to detect corruption or tampering before using it for action availability checks.
4. Integrate the guardrail into the evaluation harness so every agent action passes through it during evaluation runs.
