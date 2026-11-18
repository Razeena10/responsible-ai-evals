# Fairness Evaluation Report — responsible-ai-evals

Generated: 2026-11-18T13:40:04.567317+00:00

> **Scope note:** The groups in this evaluation are synthetic labels (Group A, Group B). They do not represent real demographic or protected populations. This is an illustrative evaluation exercise, not a real-world bias audit. Outcome-rate parity alone does not establish fairness.

---

## Dataset Description

| Attribute | Value |
|---|---|
| Total records | 120 |
| Groups | A, B |
| Records per group | 60 |
| Seed | 42 |
| Outcome definition | 1 = successful resolution, 0 = unsuccessful |

## Synthetic Data-Generation Method

The dataset was generated deterministically using a fixed random seed (42). Group A has 48 successes out of 60 records (80.0%), and Group B has 39 successes out of 60 (65.0%). Scenarios cycle through the five existing customer-support scenarios. Outcomes are shuffled deterministically. Successful outcomes use optimal resolution (5 turns, +10.5 reward). Unsuccessful outcomes are either truncated or escalated. The groups are synthetic labels — they do not represent real demographic or protected groups.

## Group-Level Outcome Counts and Rates

| Group | Total | Successful | Unsuccessful | Success Rate | Failure Rate |
|---|---|---|---|---|---|
| A | 60 | 48 | 12 | 80.0% | 20.0% |
| B | 60 | 39 | 21 | 65.0% | 35.0% |

## Fairness Comparison Metrics

| Metric | Value |
|---|---|
| Absolute difference | 15.0 pp |
| Success-rate ratio | 0.8125 |
| Higher success group | A |
| Threshold | 5.0 pp |
| Exceeds threshold | True |

## Metric Definitions and Formulas

| Metric | Formula |
|---|---|
| Success rate | successful / total (0.0 if total == 0) |
| Failure rate | unsuccessful / total (0.0 if total == 0) |
| Absolute difference (pp) | \|rate_A - rate_B\| × 100 |
| Success-rate ratio | min(rate_A, rate_B) / max(rate_A, rate_B) when max > 0; 1.0 if both 0.0; 0.0 if exactly one is 0.0 |
| Higher success group | Group with the higher rate; 'equal' if identical |
| Exceeds threshold | absolute_difference_pp > threshold_pp |

## Configurable Threshold

The illustrative threshold is set to 5.0 percentage points. This is a configurable value for flagging a potentially meaningful difference. It is not a legal or regulatory standard.

## Interpretation of Observed Results

Group A has a higher observed success rate (80.0% vs 65.0%). The absolute difference is 15.0 percentage points, which exceeds the illustrative threshold of 5.0 percentage points. This difference is illustrative and does not by itself establish unfairness or bias. Outcome-rate parity alone does not establish fairness. A small synthetic difference should not be used to label a group as disadvantaged without further investigation.

## Limitations and Assumptions

1. The groups are synthetic labels (Group A, Group B). They do not represent real demographic, protected, or marginalised populations.
2. The dataset is small (120 records) and generated deterministically. It does not capture the complexity of real customer-support interactions or real-world variation.
3. Outcome-rate parity alone does not establish fairness. Fairness requires considering context, causation, confounders, and domain-specific definitions of equity.
4. This evaluation does not test for intersectional bias, calibration across subgroups, or disparate impact under legal standards.
5. Passing or failing the illustrative threshold does not imply legal or regulatory compliance.
6. The synthetic success rates (80% and 65%) were chosen to create a meaningful difference for illustration. They were not derived from real data or real customer outcomes.

## Recommendations for Further Evaluation

1. Expand the evaluation to include more groups, subgroups, and intersectional analysis once real (de-identified) data is available.
2. Investigate the root cause of any observed disparity: is it due to scenario distribution, agent behavior, environment design, or confounding factors?
3. Consider additional fairness metrics beyond outcome parity: calibration, equalised odds, demographic parity, and procedural fairness.
4. Validate findings against real (de-identified) data before drawing any conclusions about real-world bias or fairness.
5. Engage domain experts, affected communities, and legal advisors before interpreting synthetic results as evidence of real-world bias or fairness.
