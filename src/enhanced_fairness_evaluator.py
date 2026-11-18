"""Enhanced fairness evaluation on a larger, multi-seed synthetic dataset.

The original ``fairness_evaluator`` uses 120 hand-fixed records (48/60 vs
39/60) and a single seed.  This module keeps that report untouched and runs a
separate, larger evaluation.  Everything here is synthetic: the two group
labels are arbitrary, and the numbers say nothing about real people.

Generation assumptions (fixed before any result was observed)
-------------------------------------------------------------
* 5 seeds, 1,200 records per seed (600 per group): 6,000 records total.
* Group A and Group B receive scenario labels from slightly different
  scenario mixes.  The mix difference is a deliberately documented confound,
  not a property of any real population:

      scenario          0     1     2     3     4
      group A mix     0.25  0.20  0.20  0.20  0.15
      group B mix     0.15  0.20  0.20  0.25  0.20

* Given a scenario, the outcome is Bernoulli with a scenario- and
  group-specific probability:

      scenario          0     1     2     3     4
      p(success|A)    0.90  0.60  0.80  0.70  0.88
      p(success|B)    0.72  0.78  0.80  0.86  0.88

  The gaps alternate direction (A better in scenario 0 and 4, B better in 1
  and 3, equal in 2) so that a single overall number hides opposing
  scenario-level differences.  That is the methodological point of this
  exercise; it is not evidence about any group.

Because the probabilities and mixes are chosen by hand, the resulting
difference is an artefact of the generator.  It must not be interpreted as
discrimination, nor used to claim that either group is disadvantaged.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from src.statistics_utils import (
    bootstrap_difference_interval,
    wald_difference_interval,
    wilson_interval,
)

_REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "reports"
)

SEEDS: tuple[int, ...] = (42, 7, 2026, 1234, 99)
RECORDS_PER_GROUP: int = 600
NUM_SCENARIOS: int = 5
GROUPS: tuple[str, str] = ("A", "B")

SCENARIO_MIX: dict[str, tuple[float, ...]] = {
    "A": (0.25, 0.20, 0.20, 0.20, 0.15),
    "B": (0.15, 0.20, 0.20, 0.25, 0.20),
}

SUCCESS_PROBABILITY: dict[str, tuple[float, ...]] = {
    "A": (0.90, 0.60, 0.80, 0.70, 0.88),
    "B": (0.72, 0.78, 0.80, 0.86, 0.88),
}

BOOTSTRAP_SEED = 20261117
BOOTSTRAP_RESAMPLES = 2000
THRESHOLD_PP = 5.0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_seed(seed: int) -> dict[str, np.ndarray]:
    """Generate one seed's records.

    Returns:
        ``{group: {"scenario": int array, "success": 0/1 int array}}`` for the
        two synthetic groups.  Arrays have length ``RECORDS_PER_GROUP``.
    """
    rng = np.random.default_rng(int(seed))
    records: dict[str, np.ndarray] = {}
    for group in GROUPS:
        mix = np.asarray(SCENARIO_MIX[group], dtype=float)
        scenarios = rng.choice(
            NUM_SCENARIOS, size=RECORDS_PER_GROUP, p=mix
        )
        probabilities = np.asarray(
            SUCCESS_PROBABILITY[group], dtype=float
        )[scenarios]
        successes = (rng.random(RECORDS_PER_GROUP) < probabilities).astype(
            int
        )
        records[group] = {
            "scenario": scenarios.astype(int),
            "success": successes,
        }
    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _rate_block(successes: int, total: int) -> dict[str, Any]:
    """Return a success rate with its Wilson interval."""
    low, high = wilson_interval(successes, total)
    return {
        "successes": int(successes),
        "total": int(total),
        "success_rate": (successes / total) if total else 0.0,
        "success_rate_95ci": [low, high],
    }


def _difference_block(
    successes_a: int, total_a: int, successes_b: int, total_b: int
) -> dict[str, Any]:
    """Return ``A - B`` with a Wald interval (proportion units)."""
    difference, low, high = wald_difference_interval(
        successes_a, total_a, successes_b, total_b
    )
    return {
        "difference_a_minus_b": difference,
        "difference_a_minus_b_pp": difference * 100.0,
        "difference_95ci": [low, high],
        "difference_95ci_pp": [low * 100.0, high * 100.0],
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
    }


def _pooled_arrays(
    per_seed: dict[int, dict[str, np.ndarray]]
) -> dict[str, dict[str, np.ndarray]]:
    """Concatenate scenario and success arrays across seeds."""
    return {
        group: {
            "scenario": np.concatenate(
                [per_seed[seed][group]["scenario"] for seed in SEEDS]
            ),
            "success": np.concatenate(
                [per_seed[seed][group]["success"] for seed in SEEDS]
            ),
        }
        for group in GROUPS
    }


def _scenario_table(
    pooled: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, Any]]:
    """Per-scenario group rates and differences on the pooled sample."""
    table: dict[str, dict[str, Any]] = {}
    for scenario in range(NUM_SCENARIOS):
        entry: dict[str, Any] = {}
        for group in GROUPS:
            mask = pooled[group]["scenario"] == scenario
            entry[group] = _rate_block(
                int(pooled[group]["success"][mask].sum()),
                int(mask.sum()),
            )
        entry["difference"] = _difference_block(
            entry["A"]["successes"],
            entry["A"]["total"],
            entry["B"]["successes"],
            entry["B"]["total"],
        )
        table[str(scenario)] = entry
    return table


def _composition_analysis(
    pooled: dict[str, dict[str, np.ndarray]],
    scenario_table: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Decompose the overall difference into within-scenario and mix terms.

    With ``wA_s`` and ``wB_s`` the group-specific scenario shares and ``pA_s``,
    ``pB_s`` the scenario-specific rates::

        pA - pB = sum_s wbar_s (pA_s - pB_s)          within-scenario term
                + sum_s (wA_s - wB_s) (pA_s + pB_s)/2  mix term

    where ``wbar_s = (wA_s + wB_s) / 2``.  Direct standardization to ``wbar``
    is exactly the within-scenario term, so the difference between the naive
    and standardized rates is the composition (mix) term.
    """
    total_a = sum(scenario_table[s]["A"]["total"] for s in scenario_table)
    total_b = sum(scenario_table[s]["B"]["total"] for s in scenario_table)
    total = total_a + total_b

    contributions: dict[str, dict[str, float]] = {}
    within_term = 0.0
    mix_term = 0.0
    standardized_a = 0.0
    standardized_b = 0.0
    for scenario, entry in scenario_table.items():
        w_a = entry["A"]["total"] / total_a
        w_b = entry["B"]["total"] / total_b
        w_bar = (w_a + w_b) / 2.0
        p_a = entry["A"]["success_rate"]
        p_b = entry["B"]["success_rate"]
        within_term += w_bar * (p_a - p_b)
        mix_term += (w_a - w_b) * (p_a + p_b) / 2.0
        standardized_a += w_bar * p_a
        standardized_b += w_bar * p_b
        contributions[scenario] = {
            "weight_a": w_a,
            "weight_b": w_b,
            "weight_common": w_bar,
            "rate_a": p_a,
            "rate_b": p_b,
            "difference_pp": (p_a - p_b) * 100.0,
            "weighted_contribution_pp": w_bar * (p_a - p_b) * 100.0,
        }

    naive_a = sum(
        entry["A"]["successes"] for entry in scenario_table.values()
    ) / total_a
    naive_b = sum(
        entry["B"]["successes"] for entry in scenario_table.values()
    ) / total_b

    return {
        "group_scenario_shares": {
            group: {
                scenario: (
                    scenario_table[scenario][group]["total"]
                    / (total_a if group == "A" else total_b)
                )
                for scenario in scenario_table
            }
            for group in GROUPS
        },
        "naive_difference_pp": (naive_a - naive_b) * 100.0,
        "within_scenario_difference_pp": within_term * 100.0,
        "scenario_mix_difference_pp": mix_term * 100.0,
        "decomposition_residual_pp": (
            (naive_a - naive_b) - within_term - mix_term
        )
        * 100.0,
        "standardized_rates_common_mix": {
            "A": standardized_a,
            "B": standardized_b,
        },
        "standardized_difference_pp": (standardized_a - standardized_b)
        * 100.0,
        "per_scenario": contributions,
        "interpretation": (
            "The within-scenario term is the rate difference that would "
            "remain if both groups had the same scenario mix; the mix term is "
            "the part explained by the different scenario distributions. A "
            "non-zero mix term means the overall comparison is confounded by "
            "scenario composition."
        ),
    }


def _leave_one_scenario_out(
    pooled: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, Any]]:
    """Recompute the pooled difference with each scenario excluded."""
    results: dict[str, dict[str, Any]] = {}
    for excluded in range(NUM_SCENARIOS):
        entry: dict[str, Any] = {}
        for group in GROUPS:
            mask = pooled[group]["scenario"] != excluded
            entry[group] = _rate_block(
                int(pooled[group]["success"][mask].sum()),
                int(mask.sum()),
            )
        entry["difference"] = _difference_block(
            entry["A"]["successes"],
            entry["A"]["total"],
            entry["B"]["successes"],
            entry["B"]["total"],
        )
        results[str(excluded)] = entry
    return results


def _load_original_comparison() -> dict[str, Any] | None:
    """Read the original 120-record report for comparison (never modifies it)."""
    path = os.path.join(_REPORTS_DIR, "fairness_report.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        original = json.load(handle)
    results = original.get("group_results", {})
    return {
        "source": "reports/fairness_report.json (original, unchanged)",
        "dataset_total_records": original.get("dataset", {}).get(
            "total_records"
        ),
        "group_rates": {
            group: results.get(group, {}).get("success_rate")
            for group in GROUPS
        },
        "absolute_difference_pp": original.get("comparison", {}).get(
            "absolute_difference_pp"
        ),
        "note": (
            "The original evaluation has no confidence interval and a single "
            "seed; its point estimate is shown only as a reference."
        ),
    }


def run_enhanced_fairness_evaluation() -> dict[str, Any]:
    """Run the enhanced fairness evaluation and return the report payload."""
    per_seed_raw = {seed: generate_seed(seed) for seed in SEEDS}
    per_seed: dict[str, dict[str, Any]] = {}
    for seed, records in per_seed_raw.items():
        entry: dict[str, Any] = {}
        for group in GROUPS:
            entry[group] = _rate_block(
                int(records[group]["success"].sum()),
                RECORDS_PER_GROUP,
            )
        entry["difference"] = _difference_block(
            entry["A"]["successes"],
            entry["A"]["total"],
            entry["B"]["successes"],
            entry["B"]["total"],
        )
        entry["per_scenario"] = {
            str(scenario): {
                group: _rate_block(
                    int(
                        records[group]["success"][
                            records[group]["scenario"] == scenario
                        ].sum()
                    ),
                    int(
                        (
                            records[group]["scenario"] == scenario
                        ).sum()
                    ),
                )
                for group in GROUPS
            }
            for scenario in range(NUM_SCENARIOS)
        }
        per_seed[str(seed)] = entry

    pooled = _pooled_arrays(per_seed_raw)
    pooled_rates = {
        group: _rate_block(
            int(pooled[group]["success"].sum()),
            int(pooled[group]["success"].size),
        )
        for group in GROUPS
    }
    pooled_difference = _difference_block(
        pooled_rates["A"]["successes"],
        pooled_rates["A"]["total"],
        pooled_rates["B"]["successes"],
        pooled_rates["B"]["total"],
    )
    bootstrap = bootstrap_difference_interval(
        pooled["A"]["success"],
        pooled["B"]["success"],
        seed=BOOTSTRAP_SEED,
        n_resamples=BOOTSTRAP_RESAMPLES,
    )
    pooled_difference["bootstrap"] = {
        "seed": BOOTSTRAP_SEED,
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "difference_a_minus_b": bootstrap[0],
        "difference_95ci": [bootstrap[1], bootstrap[2]],
        "difference_95ci_pp": [bootstrap[1] * 100.0, bootstrap[2] * 100.0],
        "ci_excludes_zero": bool(bootstrap[1] > 0.0 or bootstrap[2] < 0.0),
    }

    scenario_table = _scenario_table(pooled)
    composition = _composition_analysis(pooled, scenario_table)
    leave_one_out = _leave_one_scenario_out(pooled)

    seed_differences = [
        per_seed[str(seed)]["difference"]["difference_a_minus_b_pp"]
        for seed in SEEDS
    ]

    return {
        "metadata": {
            "evaluation_type": "enhanced_fairness",
            "seeds": list(SEEDS),
            "records_per_seed": RECORDS_PER_GROUP * len(GROUPS),
            "records_per_group_per_seed": RECORDS_PER_GROUP,
            "total_records": RECORDS_PER_GROUP * len(GROUPS) * len(SEEDS),
            "groups": list(GROUPS),
            "num_scenarios": NUM_SCENARIOS,
            "threshold_pp": THRESHOLD_PP,
            "threshold_meaning": (
                "Illustrative flagging threshold carried over from the "
                "original evaluation; not a legal or regulatory standard."
            ),
            "generation_model": {
                "scenario_mix": {
                    group: list(SCENARIO_MIX[group]) for group in GROUPS
                },
                "success_probability_given_scenario": {
                    group: list(SUCCESS_PROBABILITY[group])
                    for group in GROUPS
                },
                "scenario_labels": [str(s) for s in range(NUM_SCENARIOS)],
                "note": (
                    "Outcomes are Bernoulli draws from the documented "
                    "probabilities; scenario labels are drawn from the "
                    "documented group mixes. The scenario-mix difference and "
                    "the alternating per-scenario gaps are hand-chosen "
                    "assumptions of the generator."
                ),
            },
            "scope_note": (
                "Group A and Group B are synthetic labels, not real "
                "demographic or protected groups. This is not a real-world "
                "bias audit. Because the generating probabilities are chosen "
                "by hand, the observed difference is a property of this "
                "generator and must not be interpreted as discrimination or "
                "as evidence that either group is disadvantaged."
            ),
        },
        "per_seed": per_seed,
        "seed_difference_summary_pp": {
            "values": seed_differences,
            "mean": float(np.mean(seed_differences)),
            "std": float(np.std(seed_differences, ddof=1)),
            "min": float(np.min(seed_differences)),
            "max": float(np.max(seed_differences)),
            "note": (
                "Per-seed estimates vary considerably at 600 records per "
                "group; the spread shows why a single-seed result is fragile."
            ),
        },
        "pooled": {
            "groups": pooled_rates,
            "difference": pooled_difference,
            "n_records": RECORDS_PER_GROUP * len(GROUPS) * len(SEEDS),
        },
        "scenario_level": scenario_table,
        "scenario_distribution_analysis": composition,
        "leave_one_scenario_out": leave_one_out,
        "comparison_to_original_report": _load_original_comparison(),
        "limitations": [
            "Synthetic Bernoulli outcomes only; no environment, agent, or "
            "policy is simulated here.",
            "The group labels and all probabilities are arbitrary hand-chosen "
            "assumptions.",
            "The per-scenario gaps intentionally alternate direction, so an "
            "overall rate can hide opposing differences.",
            "The scenario mixes differ by group, which confounds the overall "
            "comparison; the standardized difference isolates the "
            "within-scenario part.",
            "Confidence intervals describe sampling noise in this generator, "
            "not uncertainty about any real population.",
            "Outcome-rate parity is only one fairness criterion; it does not "
            "establish fairness, and a difference does not establish bias.",
        ],
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _pp(value_pp: float) -> str:
    return f"{value_pp:+.1f}pp"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the enhanced fairness report as Markdown."""
    lines: list[str] = []
    a = lines.append
    meta = report["metadata"]

    a("# Enhanced Fairness Evaluation (Synthetic)")
    a("")
    a(
        "This report is separate from `reports/fairness_report.md`, which is "
        "preserved unchanged as the original baseline."
    )
    a("")
    a("## Scope")
    a("")
    a(f"- Seeds: {', '.join(str(s) for s in meta['seeds'])}")
    a(
        f"- {meta['records_per_seed']} records per seed "
        f"({meta['records_per_group_per_seed']} per synthetic group), "
        f"{meta['total_records']} records in total"
    )
    a(f"- Synthetic groups: {', '.join(meta['groups'])} (arbitrary labels)")
    a(
        "- Groups are **not** real demographic or protected groups, and this "
        "is **not** a real-world bias audit."
    )
    a(
        "- The outcome generator is Bernoulli with hand-chosen, documented "
        "scenario and group probabilities; the observed difference is an "
        "artefact of those choices."
    )
    a("")
    a("### Generation assumptions")
    a("")
    a(
        "| Scenario | Group A mix | Group B mix | P(success \\| A) | "
        "P(success \\| B) |"
    )
    a("|---|---|---|---|---|")
    mix = meta["generation_model"]["scenario_mix"]
    prob = meta["generation_model"]["success_probability_given_scenario"]
    for scenario in range(meta["num_scenarios"]):
        a(
            f"| {scenario} | {mix['A'][scenario]:.2f} | "
            f"{mix['B'][scenario]:.2f} | {prob['A'][scenario]:.2f} | "
            f"{prob['B'][scenario]:.2f} |"
        )
    a("")

    a("## Pooled results (all seeds)")
    a("")
    pooled = report["pooled"]
    a("| Group | Successes | Total | Success rate | 95% Wilson CI |")
    a("|---|---|---|---|---|")
    for group in meta["groups"]:
        block = pooled["groups"][group]
        a(
            f"| {group} | {block['successes']} | {block['total']} | "
            f"{_pct(block['success_rate'])} | "
            f"[{_pct(block['success_rate_95ci'][0])}, "
            f"{_pct(block['success_rate_95ci'][1])}] |"
        )
    diff = pooled["difference"]
    a("")
    a(
        f"Difference (A − B): {_pp(diff['difference_a_minus_b_pp'])} "
        f"(Wald 95% CI [{diff['difference_95ci_pp'][0]:+.1f}, "
        f"{diff['difference_95ci_pp'][1]:+.1f}]pp)"
    )
    a(
        f"Bootstrap 95% CI ({diff['bootstrap']['n_resamples']} resamples, "
        f"seed {diff['bootstrap']['seed']}): "
        f"[{diff['bootstrap']['difference_95ci_pp'][0]:+.1f}, "
        f"{diff['bootstrap']['difference_95ci_pp'][1]:+.1f}]pp"
    )
    a(
        f"Illustrative {meta['threshold_pp']:.1f}pp threshold: "
        f"{'exceeded' if abs(diff['difference_a_minus_b_pp']) > meta['threshold_pp'] else 'not exceeded'} "
        f"by the point estimate; CI excludes zero: "
        f"{'yes' if diff['ci_excludes_zero'] else 'no'}."
    )
    a("")

    a("## Per-seed results")
    a("")
    a("| Seed | A rate | B rate | Difference | Wald 95% CI |")
    a("|---|---|---|---|---|")
    for seed in meta["seeds"]:
        entry = report["per_seed"][str(seed)]
        a(
            f"| {seed} | {_pct(entry['A']['success_rate'])} | "
            f"{_pct(entry['B']['success_rate'])} | "
            f"{_pp(entry['difference']['difference_a_minus_b_pp'])} | "
            f"[{entry['difference']['difference_95ci_pp'][0]:+.1f}, "
            f"{entry['difference']['difference_95ci_pp'][1]:+.1f}]pp |"
        )
    summary = report["seed_difference_summary_pp"]
    a("")
    a(
        f"Per-seed differences: mean {summary['mean']:+.1f}pp, "
        f"sd {summary['std']:.1f}pp, range "
        f"[{summary['min']:+.1f}, {summary['max']:+.1f}]pp. "
        f"{summary['note']}"
    )
    a("")

    a("## Scenario-level results (pooled)")
    a("")
    a("| Scenario | A rate (n) | B rate (n) | Difference | Wald 95% CI |")
    a("|---|---|---|---|---|")
    for scenario, entry in report["scenario_level"].items():
        a(
            f"| {scenario} | {_pct(entry['A']['success_rate'])} "
            f"({entry['A']['total']}) | "
            f"{_pct(entry['B']['success_rate'])} "
            f"({entry['B']['total']}) | "
            f"{_pp(entry['difference']['difference_a_minus_b_pp'])} | "
            f"[{entry['difference']['difference_95ci_pp'][0]:+.1f}, "
            f"{entry['difference']['difference_95ci_pp'][1]:+.1f}]pp |"
        )
    a("")

    a("## Scenario distribution and standardization")
    a("")
    comp = report["scenario_distribution_analysis"]
    a(
        f"- Naive pooled difference: {comp['naive_difference_pp']:+.1f}pp"
    )
    a(
        f"- Within-scenario component: "
        f"{comp['within_scenario_difference_pp']:+.1f}pp"
    )
    a(
        f"- Scenario-mix component: "
        f"{comp['scenario_mix_difference_pp']:+.1f}pp"
    )
    a(
        f"- Standardized difference (common mix): "
        f"{comp['standardized_difference_pp']:+.1f}pp"
    )
    a(f"- Decomposition residual: {comp['decomposition_residual_pp']:+.1f}pp")
    a("")
    a(
        "A non-zero mix component means part of the overall difference comes "
        "from the groups receiving different scenario distributions, not from "
        "different within-scenario rates."
    )
    a("")
    a("| Scenario | Weight (common) | A − B | Weighted contribution |")
    a("|---|---|---|---|")
    for scenario, entry in comp["per_scenario"].items():
        a(
            f"| {scenario} | {entry['weight_common']:.3f} | "
            f"{_pp(entry['difference_pp'])} | "
            f"{_pp(entry['weighted_contribution_pp'])} |"
        )
    a("")

    a("## Leave-one-scenario-out")
    a("")
    a("| Excluded scenario | Difference | Wald 95% CI |")
    a("|---|---|---|")
    for scenario, entry in report["leave_one_scenario_out"].items():
        a(
            f"| {scenario} | "
            f"{_pp(entry['difference']['difference_a_minus_b_pp'])} | "
            f"[{entry['difference']['difference_95ci_pp'][0]:+.1f}, "
            f"{entry['difference']['difference_95ci_pp'][1]:+.1f}]pp |"
        )
    a("")

    original = report["comparison_to_original_report"]
    if original:
        a("## Comparison with the original 120-record report")
        a("")
        a("| Dataset | A rate | B rate | Difference |")
        a("|---|---|---|---|")
        a(
            f"| Original (120 records, seed 42) | "
            f"{_pct(original['group_rates']['A'])} | "
            f"{_pct(original['group_rates']['B'])} | "
            f"{original['absolute_difference_pp']:+.1f}pp |"
        )
        a(
            f"| Enhanced (pooled, {pooled['n_records']} records) | "
            f"{_pct(pooled['groups']['A']['success_rate'])} | "
            f"{_pct(pooled['groups']['B']['success_rate'])} | "
            f"{diff['difference_a_minus_b_pp']:+.1f}pp |"
        )
        a("")
        a(original["note"])
        a("")

    a("## Limitations")
    a("")
    for limitation in report["limitations"]:
        a(f"- {limitation}")
    a("")
    a(
        "No claim is made that the guardrail, agent, or system is fair or "
        "unfair in the real world. The differences above are properties of a "
        "synthetic generator whose probabilities were chosen by hand."
    )
    a("")
    return "\n".join(lines)


def write_reports(
    report: dict[str, Any] | None = None,
    output_dir: str = _REPORTS_DIR,
) -> tuple[str, str]:
    """Write the enhanced fairness JSON and Markdown reports."""
    payload = (
        report if report is not None else run_enhanced_fairness_evaluation()
    )
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "enhanced_fairness_report.json")
    md_path = os.path.join(output_dir, "enhanced_fairness_report.md")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(payload))
    return json_path, md_path


def main() -> int:
    payload = run_enhanced_fairness_evaluation()
    json_path, md_path = write_reports(payload)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    pooled = payload["pooled"]
    print(
        "pooled A={:.3f} B={:.3f} diff={:+.2f}pp".format(
            pooled["groups"]["A"]["success_rate"],
            pooled["groups"]["B"]["success_rate"],
            pooled["difference"]["difference_a_minus_b_pp"],
        )
    )
    print(
        "naive={:+.2f}pp within={:+.2f}pp mix={:+.2f}pp".format(
            payload["scenario_distribution_analysis"]["naive_difference_pp"],
            payload["scenario_distribution_analysis"][
                "within_scenario_difference_pp"
            ],
            payload["scenario_distribution_analysis"][
                "scenario_mix_difference_pp"
            ],
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
