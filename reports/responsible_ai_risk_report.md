# Responsible AI Risk Report — multi-turn-agent-gym

**Report scope:** Integrated Responsible AI assessment of the customer-support simulation environment, its deterministic baseline agent, the guardrail policy, and the safety and fairness evaluations.
**Evidence sources:** `reports/safety_report.json`, `reports/safety_report.md`, `reports/fairness_report.json`, `reports/fairness_report.md`, `reports/evaluation_report.json`.
**Evaluation date:** 2026-11-17 · **Seed:** 42 · **Python:** 3.14
**Status:** All 658 automated tests pass. Passing tests do **not** establish that the system is safe or fair in production.

> **Important:** This report is an engineering self-assessment. It is not a legal, regulatory, or certification assessment. It makes no claim of universal safety, universal fairness, or compliance with any standard. The fairness groups are synthetic labels, not real demographic or protected populations.

---

## 1. Executive Summary

The project implements a deterministic customer-support simulation ("multi-turn-agent-gym") together with a guardrail policy and two Responsible AI evaluations: a safety evaluation and an illustrative fairness evaluation.

- **Safety evaluation:** 41/41 predefined test cases passed across four categories (unsafe actions, repeated failures, unsupported inputs, rule-bypass attempts). Prohibited-action blocking, invalid-action blocking, unsupported-input handling, and rule-bypass blocking each reported 6/6, 7/7, 10/10, and 6/6 respectively (all 100.0% for the tested cases).
- **Fairness evaluation:** On a synthetic dataset of 120 records (60 per group), Group A resolved 48/60 (80.0%) and Group B resolved 39/60 (65.0%). The absolute difference is 15.0 percentage points (ratio 0.8125), which exceeds the illustrative 5.0 pp threshold. This is a property of deliberately constructed synthetic data, not evidence of real-world bias.
- **Environment evaluation:** The baseline agent resolved 100/100 episodes with mean reward +10.5000 and mean 5.00 turns.

These results show that the guardrail behaves as designed **for the cases tested** and that the evaluation harness is reproducible. They do not show that the system is safe or fair under all conditions. Material gaps remain: no cross-turn guardrail state, a terminating escalation fallback, a guardrail that is not exercised in the end-to-end agent loop, reliance on scenario/problem-type identification, synthetic-only fairness data, no statistical significance testing, and no production monitoring.

## 2. Project Overview

The system simulates multi-turn customer-support conversations as a Gymnasium environment. A customer reports one of five problems; the agent gathers required information, diagnoses, and proposes a solution. The project exists to:

1. Model support conversations with a structured observation/action space.
2. Define a progress-aware reward that rewards efficient, correct resolution.
3. Provide a deterministic rule-based baseline agent as a performance floor.
4. Evaluate that agent reproducibly and report metrics.
5. Apply Responsible AI evaluation to safety and fairness, and document residual risk.

Scope boundaries:

- The agent is **deterministic and rule-based**, not an LLM. There is no natural-language model, no training, and no reinforcement-learning training pipeline.
- Observations are numeric; there is no free-text input, so prompt injection and jailbreak risks are out of scope by construction rather than mitigated.
- The safety evaluation tests action-space safety; the fairness evaluation tests outcome-rate parity on synthetic groups.

## 3. System Architecture

```
BaselineSupportAgent (deterministic priority cascade)
        │ proposed action (int)
        ▼
GuardrailPolicy.evaluate(action, observation)      <-- safety layer (stateless)
        │ allowed action, or ESCALATE fallback
        ▼
CustomerSupportEnv (gymnasium.Env)
        │ reset(seed) / step(action)
        ├── support_scenarios.py  (5 scenarios, hidden root cause, user state machine)
        └── config.py             (action IDs, reward constants, thresholds)
        │
        ▼
Evaluations
        ├── demo/generate_reports.py  -> evaluation_report.json / evaluation_summary.md
        ├── src/safety_evaluator.py   -> safety_report.json / safety_report.md
        └── src/fairness_evaluator.py -> fairness_report.json / fairness_report.md
```

| Component | Responsibility |
|---|---|
| `src/config.py` | Single source of truth for action IDs, rewards, seeds, thresholds. |
| `src/support_scenarios.py` | Five scenario definitions, required fields, hidden state, customer response model. |
| `src/support_env.py` | Gymnasium environment: observations, rewards, termination/truncation. |
| `src/baseline_agent.py` | Deterministic priority-cascade policy with anti-repetition escalation. |
| `src/evaluator.py` / `src/metrics.py` | Episode harness and aggregate metrics. |
| `src/guardrails.py` | Deterministic action guardrail: validates observation, action ID, and scenario availability; blocks with ESCALATE fallback. |
| `src/safety_evaluator.py` | 41-case safety suite and JSON/Markdown safety reports. |
| `src/fairness_evaluator.py` | Synthetic two-group dataset, group metrics, comparison metrics, JSON/Markdown fairness reports. |
| `data/synthetic_fairness_data.csv` | 120-record deterministic synthetic dataset (seed 42). |
| `tests/` | 658 tests covering environment, agent, metrics, guardrails, safety, fairness, and reports. |

**Guardrail policy rules:** `ACTION_ALLOWED`, `PROHIBITED_ACTION`, `INVALID_ACTION_ID`, `MALFORMED_OBSERVATION`. The fallback action is `ESCALATE` (action 5), which is available in all five scenarios. Each decision is made independently per call; the policy holds no episode state.

## 4. Responsible AI Evaluation Methodology

The evaluations are **deterministic, offline, and reproducible** (seed 42; no LLM, network calls, or external APIs).

### Safety evaluation

Four categories of predefined cases:

| Category | Cases | Focus |
|---|---|---|
| `UNSAFE_ACTION` | 19 | Prohibited action IDs, out-of-range/non-integer/`None`/`bool` actions, plus valid-action control cases. |
| `REPEATED_FAILURE` | 6 | Stuck-agent episodes, repetition penalties, truncation, baseline resolution, stateless blocking. |
| `UNSUPPORTED_INPUT` | 10 | `None`/string/empty/missing-key/malformed observations. |
| `RULE_BYPASS` | 6 | Tampered `available_actions` masks inconsistent with scenario definitions. |
| **Total** | **41** | |

Each case records the proposed action, observation, guardrail decision, policy rule, expected/actual behavior, pass/fail, whether a risk was identified, and an explanation.

### Fairness evaluation

A deterministic synthetic dataset of 120 records (60 per group) with fields `record_id, group, scenario_id, scenario_name, turns, total_reward, outcome, termination_reason`. Outcomes are binary (1 = successful resolution, 0 = unsuccessful). Metrics:

- Success rate = successes / total (0.0 if total = 0).
- Failure rate = failures / total.
- Absolute difference (pp) = |rate_A − rate_B| × 100.
- Success-rate ratio = min(rate) / max(rate), with defined edge cases for zero rates.
- Threshold flag: `absolute_difference_pp > threshold_pp` (default 5.0 pp, configurable).

The dataset deliberately produces an unequal outcome distribution for illustration. It was not sampled from real interactions and contains no real personal or demographic data.

### Integration limits of the methodology

- The guardrail is evaluated in isolation for most safety cases; baseline agent episodes (`RF-001`, `RF-005`) explicitly run **without** the guardrail in the loop.
- The safety suite covers predefined cases only; it is not adversarial fuzzing or red-teaming.
- The fairness evaluation measures one metric family (outcome parity) on one synthetic split; it includes no confidence intervals.

## 5. Safety Evaluation Results

All values below are taken from `reports/safety_report.json`.

| Metric | Value |
|---|---|
| Total test cases | 41 |
| Passed cases | 41 |
| Failed cases | 0 |
| Block rate — prohibited actions | 6/6 = 100.0% |
| Safe-action acceptance rate (controls) | 6/6 = 100.0% |
| Invalid action block rate | 7/7 = 100.0% |
| Unsupported inputs handled safely | 10/10 = 100.0% |
| Rule-bypass block rate | 6/6 = 100.0% |
| Repeated-action risks identified | 3/6 |
| Repeated-action cases passed | 6/6 = 100.0% |

Category results: all 19 unsafe-action cases, 6 repeated-failure cases, 10 unsupported-input cases, and 6 rule-bypass cases passed.

**Reading the 3/6:** three repeated-failure scenarios (`RF-002`, `RF-003`, `RF-004`) are *designed* to demonstrate that a stuck agent can waste turns. They are marked `risk_identified: true` because the risk is real, and `passed: true` because the environment's repetition penalty and truncation fired as intended. This counts as a correctly handled risk, not a failure — but the underlying risk (an agent choosing unproductive repeats) is not eliminated.

**Reading the 100% rates:** the denominators are small and the cases are predefined (6, 6, 7, 10, 6). A 100% result means "no failures among these specific tests", not "no unsafe actions are possible". Notably, the control cases confirm the guardrail does not over-block valid actions, and the rule-bypass cases confirm the scenario definition, not the observation mask, is authoritative when they disagree.

**Failed cases:** none.

## 6. Fairness Evaluation Results

All values below are taken from `reports/fairness_report.json`.

Dataset: 120 synthetic records, 60 per group, seed 42, five support scenarios cycled deterministically.

| Group | Total | Successful | Unsuccessful | Success rate | Failure rate |
|---|---|---|---|---|---|
| A | 60 | 48 | 12 | 80.0% | 20.0% |
| B | 60 | 39 | 21 | 65.0% | 35.0% |

| Comparison metric | Value |
|---|---|
| Absolute difference | 15.0 percentage points |
| Success-rate ratio (min/max) | 0.8125 |
| Higher-success group | A |
| Threshold | 5.0 pp (illustrative, configurable) |
| Exceeds threshold | True |

**Interpretation (as reported):** Group A has a higher observed success rate (80.0% vs 65.0%); the 15.0 pp difference exceeds the illustrative 5.0 pp threshold. The difference is an artifact of how the synthetic outcomes were generated (48/60 vs 39/60 were chosen by design), and it does not by itself establish unfairness or bias. Outcome-rate parity alone does not establish fairness.

**What can and cannot be concluded:** the pipeline correctly computes and reports a disparity when one exists, and the report is explicit that the result is illustrative. Nothing here supports a statement about real demographic groups, real support outcomes, causation, or legal disparate impact.

## 7. Identified Risks and Weaknesses

The following weaknesses were recorded by the evaluations themselves and by inspection of the implementation.

1. **Repeated unproductive actions.** A poorly designed agent can repeat valid actions without progress. In `RF-002` (DIAGNOSE ×5) and `RF-003` (PROPOSE_SOLUTION ×5) the repetition penalty fired and episodes truncated with strongly negative rewards; `RF-004` (ASK_ACCOUNT_INFO ×6) behaved similarly after information was exhausted. The environment and the baseline agent's anti-repetition logic mitigate this, but the guardrail does not.
2. **Escalation terminates the episode.** Every blocked action falls back to `ESCALATE`, and escalation terminates the episode. After a guardrail intervention the system cannot continue helping the customer; the interaction ends in a handoff.
3. **No cross-turn guardrail tracking.** `GuardrailPolicy.evaluate()` is stateless. It cannot detect an agent that repeatedly attempts the same blocked action across turns; `RF-006` confirms consistent blocking per call but no accumulation or escalation logic.
4. **Reliance on scenario/problem-type identification.** Action availability is checked against `observation["problem_type"]` and the scenario definitions. Corrupted or plausible-but-wrong identifiers could cause the guardrail to validate against the wrong scenario. Malformed identifiers are blocked (`UI-008`, `UI-009`), but a semantically wrong yet valid ID is not detected.
5. **Guardrail not in the end-to-end agent loop.** Baseline episodes (`RF-001`, `RF-005`) run without the guardrail; the guardrail is exercised on isolated actions. There is no evaluation of a fully guardrailed agent episode, so interaction effects between agent policy and guardrail are untested.
6. **Synthetic fairness disparity.** The dataset reports a 15.0 pp gap that exceeds the 5.0 pp threshold. Because the rates were chosen deliberately, this is an illustration of the measurement pipeline, not a finding about a real system.
7. **No statistical significance testing.** With 60 records per group there are no confidence intervals, effect sizes with uncertainty, or hypothesis tests. The 15.0 pp gap is descriptive only.
8. **No real-world demographic data.** Groups "A" and "B" are arbitrary synthetic labels. No protected attributes, real users, or de-identified production data are involved.
9. **No production monitoring.** All evaluation is offline and deterministic. There is no runtime telemetry, drift detection, alerting, or incident review for a deployed system.

## 8. Risk Assessment

Likelihood and Impact are assessed for a hypothetical production deployment of this support agent with this guardrail. Priority reflects the need for action before such deployment. Ratings are engineering judgements, not measured probabilities.

| ID | Risk | Evidence | Likelihood | Impact | Priority | Recommended mitigation |
|---|---|---|---|---|---|---|
| R-01 | Agent repeats unproductive actions, wasting turns and degrading customer experience | `RF-002`, `RF-003`, `RF-004`: repetition penalties fired and episodes truncated; `repeated_action_risks_identified` = 3/6; weakness #4 | Medium | Medium | **Medium** | Add cross-turn repetition tracking; cap consecutive identical unproductive actions; monitor repeat rates. |
| R-02 | Guardrail fallback (ESCALATE) terminates the episode, preventing further automated help | All blocked cases (`SA-007`–`SA-012`, `UI-001`–`UI-010`, `RB-001`–`RB-006`) return ESCALATE; weakness #1; escalation terminates the environment | High | Medium | **High** | Introduce a non-terminating fallback (e.g., APOLOGIZE) for recoverable interventions; preserve handoff context; measure escalation rate. |
| R-03 | Guardrail cannot detect repeated blocked attempts across turns | `GuardrailPolicy.evaluate()` is stateless; weakness #2; `RF-006` notes stateless blocking | Medium | Medium | **Medium** | Add per-episode action history to the guardrail; define a consecutive-block policy (warn, then escalate). |
| R-04 | Wrong or corrupted `problem_type` causes availability checks against the wrong scenario | Weakness #3; availability check uses `observation["problem_type"]`; `UI-008`/`UI-009` block invalid values but not valid-yet-wrong ones | Low | High | **High** | Cross-validate `problem_type` against `customer_issue` and the scenario registry; log and block mismatches; add integrity checks. |
| R-05 | Synthetic 15.0 pp disparity is misread as evidence of real bias (or of fairness) | Fairness report: 80.0% vs 65.0%, diff 15.0 pp, ratio 0.8125, exceeds 5.0 pp threshold | High | Low | **Medium** | Label results as illustrative everywhere; investigate root causes before interpreting; never generalize to real groups. |
| R-06 | No statistical significance testing; disparity may be noise or may be under-powered | Fairness report has no confidence intervals; n = 60 per group; threshold is illustrative | High | Medium | **Medium** | Add Wilson/bootstrap confidence intervals, effect sizes, and power analysis; pre-register thresholds. |
| R-07 | Absence of real demographic data prevents any real fairness conclusion | Dataset groups are "A"/"B"; fairness limitations #1; synthetic seed 42 | High | High | **High** | Obtain governance approval and de-identified data; define protected attributes and lawful basis; conduct an impact assessment. |
| R-08 | No production monitoring means regressions, drift, and incidents go undetected | Safety limitations #4; all evaluations offline/deterministic; no telemetry in the codebase | High | High | **High** | Add runtime logging, guardrail-block and escalation metrics, drift alerts, and an incident-review process. |
| R-09 | Passing 658 tests and 41/41 safety cases creates false assurance | Safety limitations #3; 100% rates rest on 6–19 predefined cases per category | Medium | High | **High** | State scope caveats prominently; add adversarial/red-team testing; stage rollout behind human oversight; track coverage, not just pass rate. |
| R-10 | Guardrail is not exercised in the end-to-end agent loop, so integration behavior is untested | `RF-001`/`RF-005` policy rule "N/A (agent-driven, no guardrail in loop)"; safety mitigation #4 | High | Medium | **High** | Wrap every agent action with the guardrail in the evaluation harness and add end-to-end guardrailed episode tests. |

**Aggregate view:** two risks are High/High (R-07, R-08) and several are High priority because of governance or integration gaps rather than observed test failures. No risk in this register is eliminated by the current test results.

## 9. Recommended Mitigations

Ordered by the risk register; the first group addresses structural gaps, the second improves measurement, and the third addresses governance.

**Structural / engineering**

1. **R-02:** Replace unconditional ESCALATE fallback with a graded response: retry-safe fallback (APOLOGIZE) for recoverable cases, escalation only when the observation is untrustworthy or the action is genuinely prohibited. Record why escalation occurred.
2. **R-03, R-01:** Give the guardrail per-episode state: track consecutive blocked/unproductive actions and apply a defined policy (warning, alternative action, then escalation). Coordinate with the environment's existing repetition penalty so the two do not double-penalize.
3. **R-04:** Validate `problem_type` against `customer_issue` and the scenario registry before availability checks; treat disagreement as a malformed observation. Add tests for valid-but-wrong IDs.
4. **R-10:** Integrate the guardrail into the evaluation harness so every action from the agent passes through it, and add end-to-end tests for guardrailed episodes. Re-run the safety suite through this path.

**Measurement**

5. **R-06:** Add confidence intervals (Wilson for proportions, bootstrap for differences) and effect sizes to the fairness report; distinguish descriptive from inferential results.
6. **R-05:** Keep synthetic results clearly labelled, and add a mandatory root-cause step (scenario mix, agent policy, environment design, confounders) before any disparity is interpreted.
7. **R-09:** Expand safety testing beyond predefined cases: adversarial/fuzz observations, mutated masks, and randomized action sequences. Report coverage and per-category denominators alongside pass rates.

**Governance**

8. **R-07:** Do not collect or infer demographic data without approval. If real fairness evaluation is pursued, define the lawful basis, de-identification, retention, and review process first.
9. **R-08:** Before any deployment, define monitoring: guardrail block rate, escalation rate, resolution rate by scenario, reward distribution, and drift alerts, with an on-call review path.
10. **R-02, R-09:** Document the human-handoff contract (what context reaches the human agent) and require human oversight for escalated and low-confidence cases.

## 10. Known Limitations

**Safety evaluation (from `reports/safety_report.json`)**

1. Tests a deterministic, discrete-action support agent — not an LLM. Natural-language prompt injection, jailbreaks, and LLM-specific safety concerns are not tested.
2. Covers predefined scenarios and cases only; no stochastic scenarios, production edge cases, or novel attack vectors.
3. Passing all cases confirms the guardrail works as designed **for the tested cases**; it does not prove safety under all conditions.
4. Deterministic and reproducible; does not account for random failures, race conditions, or production environment issues.

**Fairness evaluation (from `reports/fairness_report.json`)**

5. Groups are synthetic labels and do not represent real demographic, protected, or marginalised populations.
6. The dataset is small (120 records) and deterministic; it does not capture real support complexity or variation.
7. Outcome-rate parity alone does not establish fairness; context, causation, confounders, and domain definitions of equity are missing.
8. No intersectional analysis, calibration across subgroups, or disparate-impact analysis under legal standards.
9. Passing or failing the illustrative threshold implies no legal or regulatory compliance.
10. The 80%/65% rates were chosen for illustration and were not derived from real data or outcomes.

**System / report limitations**

11. The environment evaluation reports 100% resolution by a deterministic agent on hand-crafted scenarios; this reflects a narrow, solvable benchmark rather than general capability.
12. The fairness report contains no confidence intervals or significance tests.
13. This risk report is a static document maintained manually; it is not auto-generated from the JSON reports, so it must be reviewed whenever evaluations or code change.
14. No production deployment, monitoring, or incident data exists; all likelihood/impact ratings are engineering judgements.

## 11. Responsible AI Considerations

- **Transparency:** All metrics, formulas, thresholds, seeds, and test cases are published in machine-readable JSON and Markdown. Results state what was and was not tested.
- **Non-maleficence:** The guardrail blocks prohibited and invalid actions and falls back to a safe action; it does not over-block valid actions in the control cases. However, a terminating fallback can still leave a customer unresolved.
- **Fairness:** The project measures one narrow notion (outcome parity) on synthetic groups. It explicitly avoids claiming that parity equals fairness and avoids claims about real protected groups.
- **Accountability:** Ownership of thresholds, escalation policy, and any future use of real data is not defined in code; this must be assigned before deployment.
- **Human oversight:** ESCALATE is the designed path to a human. Its termination behavior and handoff payload need product decisions, not just code.
- **Privacy:** No real personal data is used. Any future use of real support transcripts would require a lawful basis, de-identification, and retention controls.
- **Governance:** Passing tests is a development gate, not an assurance certificate. Release decisions should require review of this risk register, not only a green test run.
- **Communication:** Reports consistently label synthetic results and avoid 100% safety/fairness claims. Downstream summaries should preserve those labels.

## 12. Conclusion

The project delivers a reproducible safety and fairness evaluation pipeline with honest reporting. The safety suite passes 41/41 predefined cases, with no failed cases, and the fairness pipeline correctly detects and reports a 15.0 pp synthetic disparity against a 5.0 pp illustrative threshold. The full test suite (658 tests) passes.

The evidence supports a narrow conclusion: **the guardrail and reporting pipeline behave as designed for the specific cases tested.** It does not support claims that the system is safe, fair, unbiased, or compliant in production. The highest-priority gaps are the absence of real demographic data (R-07), the absence of production monitoring (R-08), the terminating escalation fallback (R-02), reliance on scenario identification (R-04), the guardrail's lack of cross-turn state (R-03), and the guardrail not being exercised end-to-end (R-10). These are engineering and governance work items, not test failures.

---

## Appendix A — Conceptual Mapping to AWS SageMaker Responsible AI Capabilities

This appendix explains how the project's work relates **conceptually** to AWS Responsible AI tooling. It is a mapping of ideas, not an integration.

**AWS provides a set of capabilities rather than a single "SageMaker Responsible AI" product.** Relevant examples include Amazon SageMaker Clarify (bias metrics and explainability), SageMaker Model Monitor (data quality, model quality, bias drift, and feature-attribution drift), SageMaker Model Registry (model cataloguing and approval), SageMaker Model Cards (model documentation), SageMaker Pipelines and Experiments (orchestration and run tracking), and Amazon Augmented AI (human review). Guardrails for generative AI workloads are associated with Amazon Bedrock Guardrails, not SageMaker.

| Capability area | What this project implements locally | Relationship to AWS |
|---|---|---|
| Safety evaluation | `src/safety_evaluator.py`: 41 deterministic cases in four categories, with pass/fail, policy rule, and risk flags | **Conceptually related** to structured model evaluation (e.g., Amazon Bedrock model evaluation for foundation models). Different mechanism, different scope. |
| Guardrails | `src/guardrails.py`: deterministic action validation on observation shape, action ID, and scenario availability, with ESCALATE fallback | **Conceptually related** to Amazon Bedrock Guardrails and to pre-inference validation patterns. This is a custom Python policy over a discrete action space — not Bedrock Guardrails, and not a content filter for text. |
| Bias and fairness monitoring | `src/fairness_evaluator.py`: offline, one-shot outcome-parity comparison on synthetic data | **Conceptually related** to SageMaker Clarify (pre-training/post-training bias metrics) and SageMaker Model Monitor (bias drift). This project has no continuous monitoring, no real attributes, and no Clarify integration. |
| Model evaluation and governance | `pytest` suite, JSON/Markdown reports, seeded reproducibility, documented limitations | **Conceptually related** to SageMaker Experiments (run tracking), Pipelines (repeatable evaluation), Model Registry (approval gates), and Model Cards (documentation). None of these services is used. |
| Human oversight / escalation | `ESCALATE` action and guardrail fallback to a human agent | **Conceptually related** to human-review workflows such as Amazon Augmented AI. No human-review service is configured; the handoff is simulated inside the environment. |

### What is *not* done

- No AWS account, credentials, IAM roles, S3 buckets, endpoints, or SDK calls are used anywhere in this repository.
- No model is trained, registered, deployed, or monitored in SageMaker.
- No SageMaker Clarify job, Model Monitor schedule, Pipeline, Experiment, Model Registry entry, or Model Card exists.
- No Amazon Bedrock Guardrail is configured; the local guardrail is unrelated software.
- No claim of AWS integration, deployment, certification, or production readiness is made.

The mapping is useful for planning: a future production system could place a comparable evaluation in SageMaker Pipelines, use Clarify for bias metrics on approved data, publish model documentation through Model Cards, gate releases in Model Registry, and monitor drift with Model Monitor. Those are proposals, not completed work.

---

*Prepared from `reports/safety_report.json` and `reports/fairness_report.json` generated on 2026-11-17 (seed 42). Metrics in this report must be re-verified if the evaluations are re-run with different code or data.*
