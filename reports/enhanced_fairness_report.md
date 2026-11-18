# Enhanced Fairness Evaluation (Synthetic)

This report is separate from `reports/fairness_report.md`, which is preserved unchanged as the original baseline.

## Scope

- Seeds: 42, 7, 2026, 1234, 99
- 1200 records per seed (600 per synthetic group), 6000 records in total
- Synthetic groups: A, B (arbitrary labels)
- Groups are **not** real demographic or protected groups, and this is **not** a real-world bias audit.
- The outcome generator is Bernoulli with hand-chosen, documented scenario and group probabilities; the observed difference is an artefact of those choices.

### Generation assumptions

| Scenario | Group A mix | Group B mix | P(success \| A) | P(success \| B) |
|---|---|---|---|---|
| 0 | 0.25 | 0.15 | 0.90 | 0.72 |
| 1 | 0.20 | 0.20 | 0.60 | 0.78 |
| 2 | 0.20 | 0.20 | 0.80 | 0.80 |
| 3 | 0.20 | 0.25 | 0.70 | 0.86 |
| 4 | 0.15 | 0.20 | 0.88 | 0.88 |

## Pooled results (all seeds)

| Group | Successes | Total | Success rate | 95% Wilson CI |
|---|---|---|---|---|
| A | 2296 | 3000 | 76.5% | [75.0%, 78.0%] |
| B | 2450 | 3000 | 81.7% | [80.2%, 83.0%] |

Difference (A − B): -5.1pp (Wald 95% CI [-7.2, -3.1]pp)
Bootstrap 95% CI (2000 resamples, seed 20261117): [-7.2, -3.3]pp
Illustrative 5.0pp threshold: exceeded by the point estimate; CI excludes zero: yes.

## Per-seed results

| Seed | A rate | B rate | Difference | Wald 95% CI |
|---|---|---|---|---|
| 42 | 76.3% | 82.3% | -6.0pp | [-10.6, -1.4]pp |
| 7 | 78.0% | 80.5% | -2.5pp | [-7.1, +2.1]pp |
| 2026 | 74.7% | 81.0% | -6.3pp | [-11.0, -1.6]pp |
| 1234 | 78.7% | 82.7% | -4.0pp | [-8.5, +0.5]pp |
| 99 | 75.0% | 81.8% | -6.8pp | [-11.5, -2.2]pp |

Per-seed differences: mean -5.1pp, sd 1.8pp, range [-6.8, -2.5]pp. Per-seed estimates vary considerably at 600 records per group; the spread shows why a single-seed result is fragile.

## Scenario-level results (pooled)

| Scenario | A rate (n) | B rate (n) | Difference | Wald 95% CI |
|---|---|---|---|---|
| 0 | 89.2% (729) | 71.4% (419) | +17.8pp | [+12.9, +22.7]pp |
| 1 | 58.7% (589) | 79.5% (650) | -20.8pp | [-25.8, -15.8]pp |
| 2 | 77.4% (619) | 79.7% (586) | -2.3pp | [-6.9, +2.3]pp |
| 3 | 69.4% (618) | 87.1% (735) | -17.7pp | [-22.0, -13.3]pp |
| 4 | 88.1% (445) | 86.4% (610) | +1.7pp | [-2.4, +5.8]pp |

## Scenario distribution and standardization

- Naive pooled difference: -5.1pp
- Within-scenario component: -5.0pp
- Scenario-mix component: -0.1pp
- Standardized difference (common mix): -5.0pp
- Decomposition residual: -0.0pp

A non-zero mix component means part of the overall difference comes from the groups receiving different scenario distributions, not from different within-scenario rates.

| Scenario | Weight (common) | A − B | Weighted contribution |
|---|---|---|---|
| 0 | 0.191 | +17.8pp | +3.4pp |
| 1 | 0.207 | -20.8pp | -4.3pp |
| 2 | 0.201 | -2.3pp | -0.5pp |
| 3 | 0.225 | -17.7pp | -4.0pp |
| 4 | 0.176 | +1.7pp | +0.3pp |

## Leave-one-scenario-out

| Excluded scenario | Difference | Wald 95% CI |
|---|---|---|
| 0 | -10.9pp | [-13.2, -8.5]pp |
| 1 | -1.4pp | [-3.6, +0.8]pp |
| 2 | -5.8pp | [-8.1, -3.5]pp |
| 3 | -1.5pp | [-3.9, +0.8]pp |
| 4 | -5.9pp | [-8.3, -3.6]pp |

## Comparison with the original 120-record report

| Dataset | A rate | B rate | Difference |
|---|---|---|---|
| Original (120 records, seed 42) | 80.0% | 65.0% | +15.0pp |
| Enhanced (pooled, 6000 records) | 76.5% | 81.7% | -5.1pp |

The original evaluation has no confidence interval and a single seed; its point estimate is shown only as a reference.

## Limitations

- Synthetic Bernoulli outcomes only; no environment, agent, or policy is simulated here.
- The group labels and all probabilities are arbitrary hand-chosen assumptions.
- The per-scenario gaps intentionally alternate direction, so an overall rate can hide opposing differences.
- The scenario mixes differ by group, which confounds the overall comparison; the standardized difference isolates the within-scenario part.
- Confidence intervals describe sampling noise in this generator, not uncertainty about any real population.
- Outcome-rate parity is only one fairness criterion; it does not establish fairness, and a difference does not establish bias.

No claim is made that the guardrail, agent, or system is fair or unfair in the real world. The differences above are properties of a synthetic generator whose probabilities were chosen by hand.
