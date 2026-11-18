# multi-turn-agent-gym

A custom [Gymnasium](https://gymnasium.farama.org/) environment that simulates a customer-support agent resolving user problems over multiple conversation turns. The project includes a deterministic rule-based baseline agent, a reproducible evaluation harness, automated metrics, a deterministic action guardrail, safety and fairness evaluations, and a full Pytest test suite.

> **Scope note:** This is a *simulated* multi-turn Gymnasium environment with a rule-based baseline agent. No language model was trained using reinforcement learning. The project demonstrates environment engineering, agent design, and evaluation methodology — it is not an RL training pipeline.

---

## Business Problem and Motivation

Customer-support conversations are multi-step processes. An agent must gather the right information, diagnose the root cause, and propose a solution — often across several exchanges. Modelling this as a Gymnasium environment lets us:

- Define a clear reward signal for efficient, correct resolution.
- Measure baseline performance before training learned agents.
- Tune reward shaping and termination conditions in isolation.
- Provide a reproducible benchmark for comparing future policies.

---

## Project Objectives

1. Implement a custom `gymnasium.Env` that models multi-turn customer-support conversations.
2. Define a structured observation and discrete action space without NLP dependencies.
3. Engineer a progress-aware reward function that incentivises efficient resolution.
4. Build a deterministic rule-based baseline agent to establish a performance floor.
5. Evaluate the agent over 100 episodes and report aggregate and per-scenario metrics.
6. Apply a deterministic guardrail and run safety and fairness evaluations against it.
7. Validate every component with a comprehensive Pytest suite (658 tests).

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| RL Framework | Gymnasium 1.4.0 |
| Numerical Ops | NumPy 2.5.4 |
| Testing | Pytest 9.1.1 |

---

## Project Structure

```
multi-turn-agent-gym/
├── requirements.txt           # Pinned runtime dependencies
├── requirements-dev.txt       # Pinned test dependencies
├── pytest.ini                 # Pytest configuration
│
├── src/
│   ├── config.py              # All constants — actions, rewards, thresholds
│   ├── support_scenarios.py   # Scenario definitions, conversation state, user state machine
│   ├── support_env.py         # CustomerSupportEnv (gymnasium.Env subclass)
│   ├── baseline_agent.py      # BaselineSupportAgent — deterministic rule-based policy
│   ├── evaluator.py           # evaluate_agent() harness
│   ├── metrics.py             # compute_metrics(), compute_scenario_breakdown()
│   ├── guardrails.py          # GuardrailPolicy — deterministic action validation
│   ├── safety_evaluator.py    # 41-case safety suite + JSON/Markdown reports
│   └── fairness_evaluator.py  # Synthetic two-group fairness evaluation + reports
│
├── data/
│   └── synthetic_fairness_data.csv  # 120 synthetic records, seed 42
│
├── tests/
│   ├── test_environment.py    # Environment API, rewards, termination
│   ├── test_agent.py          # Agent policy, robustness, integration
│   ├── test_evaluator.py      # Evaluation loop, edge cases
│   ├── test_metrics.py        # Metrics computation
│   ├── test_wave6.py          # Comprehensive edge cases, integration
│   ├── test_wave7.py          # Demo, report generation, README checks
│   ├── test_guardrails.py     # Guardrail policy and rule coverage
│   ├── test_safety.py         # Safety evaluator and report consistency
│   └── test_fairness.py       # Fairness evaluator, dataset, report consistency
│
├── demo/
│   ├── run_demo.py            # Interactive demo (single episode / scenario / batch)
│   └── generate_reports.py    # Generates evaluation_report.json + evaluation_summary.md
│
└── reports/
    ├── evaluation_report.json        # Full JSON evaluation output
    ├── evaluation_summary.md         # Human-readable evaluation summary
    ├── safety_report.json            # Safety evaluation output
    ├── safety_report.md              # Human-readable safety summary
    ├── fairness_report.json          # Fairness evaluation output
    ├── fairness_report.md            # Human-readable fairness summary
    └── responsible_ai_risk_report.md # Integrated Responsible AI risk report
```

---

## Architecture and Component Responsibilities

```
BaselineSupportAgent
      │ select_action(observation) → int
      ▼
GuardrailPolicy (src/guardrails.py)   ← deterministic action filter
      │ allowed action, or ESCALATE fallback
      ▼
CustomerSupportEnv (gymnasium.Env)
      │ reset(seed) → (obs, info)
      │ step(action) → (obs, reward, terminated, truncated, info)
      │
      ├── support_scenarios.py
      │     ├── Scenario (frozen dataclass, 5 definitions)
      │     ├── ConversationState (mutable, hidden state inside)
      │     ├── ScenarioRegistry (samples via seeded RNG)
      │     └── UserStateMachine (processes actions → UserResponse)
      │
      └── config.py (all reward constants, action IDs, defaults)

evaluate_agent(env, agent, num_episodes, seed) → list[dict]
compute_metrics(records) → dict
compute_scenario_breakdown(records) → dict
```

**`src/config.py`** — single source of truth for all constants: action IDs, message types, reward values, frustration thresholds, defaults.

**`src/support_scenarios.py`** — scenario definitions (customer issue, required fields, ask mappings), hidden state (root cause, diagnosis flags), and the user state machine that simulates realistic customer responses.

**`src/support_env.py`** — the Gymnasium environment. Calls `UserStateMachine`, computes rewards, checks termination/truncation, and builds agent-visible observations. Hidden state is never included in observations or info dicts.

**`src/baseline_agent.py`** — deterministic priority-cascade policy. Uses observations only — never accesses environment internals or hidden state.

**`src/evaluator.py`** — runs N episodes, resets both env and agent each time, uses additive per-episode seeding (`seed + episode_id`), includes a defensive max-steps safeguard.

**`src/metrics.py`** — aggregates episode records into counts, rates, reward statistics, and per-scenario breakdowns using NumPy. Returns plain Python types for JSON serialisation.

**`src/guardrails.py`** — deterministic `GuardrailPolicy`. Validates the observation shape, the action ID, and whether the action is available for the scenario, then allows the action or substitutes the safe fallback (`ESCALATE`). Stateless: each call is evaluated independently.

**`src/safety_evaluator.py`** — runs 41 predefined safety cases in four categories (unsafe actions, repeated failures, unsupported inputs, rule-bypass attempts) and writes the safety reports.

**`src/fairness_evaluator.py`** — generates a reproducible synthetic two-group dataset, computes group-level and comparison metrics, and writes the fairness reports. The groups are synthetic labels, not real demographic groups.

---

## Environment Design

### Observation Space (`spaces.Dict`)

| Field | Space | Description |
|---|---|---|
| `customer_issue` | `Discrete(5)` | Scenario ID (agent-visible label) |
| `collected_information` | `MultiBinary(4)` | Which required fields have been gathered |
| `missing_information` | `MultiBinary(4)` | Inverse of `collected_information` |
| `turn_count` | `Discrete(16)` | Current turn number (0–max_turns) |
| `frustration` | `Box(0.0, 1.0)` | Customer frustration level |
| `previous_customer_response` | `Discrete(8)` | Customer's last message category |
| `available_actions` | `MultiBinary(7)` | Scenario-relevant action mask |
| `problem_type` | `Discrete(5)` | Scenario ID (same as `customer_issue`) |

Hidden state — `hidden_issue`, `diagnosed`, `solution_proposed`, `solution_rejected`, `last_action` — is never included in observations or info dicts.

### Action Space (`spaces.Discrete(7)`)

| ID | Name | Description |
|---|---|---|
| 0 | `ASK_ACCOUNT_INFO` | Request identity / account details |
| 1 | `ASK_BILLING_INFO` | Request billing / payment details |
| 2 | `ASK_TECHNICAL_INFO` | Request technical details (device, error, etc.) |
| 3 | `DIAGNOSE` | Attempt to identify the root cause |
| 4 | `PROPOSE_SOLUTION` | Propose a resolution to the customer |
| 5 | `ESCALATE` | Hand off to a human agent |
| 6 | `APOLOGIZE` | Apologise to reduce frustration |

### Reward Structure

| Event | Reward |
|---|---|
| Correct resolution (all info + diagnosed + accepted) | +10.0 |
| Gathering a new required information field | +1.0 |
| Each turn taken (efficiency pressure) | −0.5 |
| Proposing an incorrect solution | −3.0 |
| **Unproductive** repeated action (same action, no new info gathered) | −2.0 |
| Escalation | +2.0 |
| Turn limit reached without resolution | −5.0 |
| Apologise with frustration > 0.5 | +0.5 |
| Apologise with frustration ≤ 0.5 | 0.0 |

**Progress-aware repetition penalty:** The −2.0 penalty only fires when the same action is repeated **and no new information was gathered**. Productive repetitions — such as `ASK_ACCOUNT_INFO` being called three times in the Password Reset scenario to collect three different fields — are not penalised. APOLOGIZE is always exempt.

**Optimal episode reward:** 3 info turns + 1 diagnose + 1 propose = 5 turns.
`3 × (1.0 − 0.5) + (0.0 − 0.5) + (10.0 − 0.5) = +10.5`

### Episode Termination and Truncation

| Condition | Flag | `termination_reason` |
|---|---|---|
| Agent proposes correct solution (all info collected + diagnosed) | `terminated=True` | `"resolved"` |
| Agent escalates | `terminated=True` | `"escalated"` |
| Turn limit reached without resolution | `truncated=True` | `"truncated"` |

`terminated` and `truncated` are **never both True**. Resolution/escalation are checked before truncation, so resolving on the final allowed turn correctly returns `terminated=True`.

### Turn Limit

Default `max_turns=15`. Configurable at construction: `CustomerSupportEnv(max_turns=N)`.

---

## Five Customer-Support Scenarios

| ID | Scenario | Customer Issue | Required Fields |
|---|---|---|---|
| 0 | Internet Connection Problem | "My internet keeps dropping every few minutes" | account_id, device_type, connection_type |
| 1 | Router Malfunction | "My router won't connect, the light is red" | account_id, router_model, error_indicator |
| 2 | Account Login Problem | "I can't log in, it says invalid credentials" | account_id, email, login_error |
| 3 | Password Reset Request | "I forgot my password and need to reset it" | account_id, email, security_question |
| 4 | Billing Issue | "I was charged twice for my last bill" | account_id, last_charge, payment_method |

Each scenario has a **hidden root cause** (e.g., "DSL line degradation") and a **required information path** that the agent must follow without knowing the root cause. The agent only observes the customer's messages and the collected-information vector.

---

## Baseline Agent Decision-Making Strategy

The `BaselineSupportAgent` follows a deterministic **priority cascade**:

1. **Frustration check** — if `frustration > 0.7`, apologise to de-escalate.
2. **Rejection check** — if the customer rejected the last solution, escalate.
3. **Information gathering** — ask for the first uncollected required field using the correct `ASK_*` action.
4. **Diagnosis** — once all info is collected, diagnose.
5. **Propose solution** — once diagnosed, propose the solution.
6. **Fallback** — escalate.

**Anti-repetition logic:** If the same non-APOLOGIZE action would repeat for the third consecutive time and can no longer gather new information, the agent escalates instead of looping indefinitely. This is separate from the environment's penalty, which fires on unproductive repeats.

The agent is **fully deterministic** — identical observations always produce identical actions. It never accesses the environment object, hidden state, or private attributes.

---

## Evaluation Methodology

- **Harness:** `evaluate_agent(env, agent, num_episodes, seed)`
- **Seeding:** Each episode uses `seed + episode_id` for independent, reproducible results.
- **Records:** Each episode produces `episode_id, scenario_id, total_reward, number_of_turns, success, terminated, truncated, termination_reason`.
- **Safeguard:** A defensive `max_steps` parameter prevents infinite loops.
- **Metrics:** `compute_metrics()` aggregates counts, rates, and reward statistics using NumPy. `compute_scenario_breakdown()` produces per-scenario results.

---

## Evaluation Results (100 Episodes, Seed=42)

Results generated by running `python -m demo.generate_reports`. No numbers are hardcoded.

### Aggregate Metrics

| Metric | Value |
|---|---|
| Total episodes | 100 |
| Successful resolutions | 100 |
| Success rate | **100.0%** |
| Mean reward | **+10.5000** |
| Std reward | 0.0000 |
| Min reward | +10.5000 |
| Max reward | +10.5000 |
| Mean turns | **5.00** |
| Terminated episodes | 100 |
| Truncated episodes | 0 |

### Per-Scenario Results

| Scenario | Episodes | Resolved | Rate | Mean Reward | Mean Turns |
|---|---|---|---|---|---|
| Internet Connection Problem | 20 | 20 | 100% | +10.50 | 5.0 |
| Router Malfunction | 13 | 13 | 100% | +10.50 | 5.0 |
| Account Login Problem | 27 | 27 | 100% | +10.50 | 5.0 |
| Password Reset Request | 23 | 23 | 100% | +10.50 | 5.0 |
| Billing Issue | 17 | 17 | 100% | +10.50 | 5.0 |

The baseline agent achieves 100% success across all scenarios in exactly 5 turns each. The uniform +10.50 reward confirms the progress-aware penalty is working correctly — productive repeated ASK actions in scenarios 2, 3, and 4 are not penalised.

---

## Responsible AI Evaluation Suite

This project includes a guardrail and two offline, deterministic Responsible AI evaluations. All results below come from the committed JSON reports; no numbers are hardcoded.

### Guardrail policy (`src/guardrails.py`)

`GuardrailPolicy.evaluate(action, observation)` checks, in order:

1. The observation is well-formed (dict with the expected keys and ranges).
2. The action is an integer in `[0, 7)` (rejecting floats, strings, `None`, and `bool`).
3. The action is available for the observation's scenario, cross-checking the observation mask against the scenario definition.

If any check fails, the action is blocked and replaced with the safe fallback `ESCALATE`. The policy is fully deterministic and stateless — it does not track actions across turns.

### Safety evaluation

```bash
python -m src.safety_evaluator
```

Runs 41 predefined cases in four categories and writes `reports/safety_report.json` and `reports/safety_report.md`. Results (seed 42):

| Metric | Value |
|---|---|
| Total test cases | 41 |
| Passed / failed | 41 / 0 |
| Prohibited actions blocked | 6/6 = 100.0% |
| Valid actions accepted (controls) | 6/6 = 100.0% |
| Invalid actions blocked | 7/7 = 100.0% |
| Unsupported inputs handled safely | 10/10 = 100.0% |
| Rule-bypass attempts blocked | 6/6 = 100.0% |
| Repeated-action risks identified | 3/6 |
| Repeated-action cases passed | 6/6 = 100.0% |

These 100% rates apply only to the predefined cases. The evaluation also records four weaknesses: the ESCALATE fallback terminates the episode; the guardrail has no cross-turn state; defense-in-depth depends on `problem_type` accuracy; and repetition safeguards live in the environment rather than the guardrail.

### Fairness evaluation

```bash
python -m src.fairness_evaluator
```

Generates or loads the deterministic synthetic dataset (`data/synthetic_fairness_data.csv`, 120 records, seed 42) and writes `reports/fairness_report.json` and `reports/fairness_report.md`. Results:

| Group | Total | Successful | Failed | Success rate |
|---|---|---|---|---|
| A | 60 | 48 | 12 | 80.0% |
| B | 60 | 39 | 21 | 65.0% |

| Comparison metric | Value |
|---|---|
| Absolute difference | 15.0 percentage points |
| Success-rate ratio (min/max) | 0.8125 |
| Higher-success group | A |
| Threshold (illustrative) | 5.0 pp |
| Exceeds threshold | True |

The groups are **synthetic labels**, not real demographic or protected populations. The outcome rates were chosen to illustrate the measurement pipeline, so the 15.0 pp gap is not evidence of real-world bias. Outcome-rate parity alone does not establish fairness, and these results imply no legal or regulatory compliance.

### Integrated risk report

`reports/responsible_ai_risk_report.md` integrates the safety and fairness results into a single Responsible AI assessment. It contains an executive summary, methodology, both result sets, a risk register with likelihood/impact/priority ratings, mitigations, preserved limitations, and a conceptual (not implemented) mapping to AWS SageMaker Responsible AI capabilities.

The report is a **maintained document**, not auto-generated. To refresh its evidence, re-run the three commands above and review the report against the regenerated JSON; update any changed metrics by hand.

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd multi-turn-agent-gym

# Install dependencies
pip install -r requirements-dev.txt
```

---

## Commands

```bash
# Run the interactive demo
python -m demo.run_demo

# Generate JSON and Markdown evaluation reports
python -m demo.generate_reports

# Generate reports with custom settings
python -m demo.generate_reports --episodes 50 --seed 123

# Run the guardrail, safety, and fairness tests
python -m pytest tests/test_guardrails.py -v
python -m pytest tests/test_safety.py -v
python -m pytest tests/test_fairness.py -v

# Run the safety evaluation (regenerates safety_report.json/.md)
python -m src.safety_evaluator

# Run the fairness evaluation (regenerates fairness_report.json/.md)
python -m src.fairness_evaluator

# Run the full test suite
python -m pytest

# Run with verbose output
python -m pytest -v
```

**Windows console note:** the demo report generator prints Unicode box characters. If your terminal uses a legacy code page (cp1252), run it with `PYTHONIOENCODING=utf-8 python -m demo.generate_reports` to avoid a `UnicodeEncodeError` after the reports are written. The safety and fairness CLIs use ASCII output and do not need this.

---

## Reports

### `reports/evaluation_report.json`

A machine-readable JSON file with:
- `generated_at` — UTC timestamp.
- `environment` / `agent` — component names.
- `config` — episodes, seed, max_turns, num_scenarios.
- `metrics` — all aggregate metrics (12 keys).
- `scenario_breakdown` — per-scenario dict (keys "0"–"4").
- `episodes` — full per-episode list (episode_id, scenario_id, total_reward, number_of_turns, success, terminated, truncated, termination_reason).

### `reports/evaluation_summary.md`

A human-readable Markdown report with the same content in table form. Suitable for inspection in GitHub or any Markdown viewer.

### `reports/safety_report.json` / `reports/safety_report.md`

Machine-readable and human-readable safety results: 41 test cases, aggregate rates with explicit numerators/denominators, failed cases, weaknesses, limitations, and mitigations.

### `reports/fairness_report.json` / `reports/fairness_report.md`

Machine-readable and human-readable fairness results: dataset description and generation method, per-group counts and rates, comparison metrics and formulas, threshold meaning, interpretation, limitations, and recommendations. The Markdown values match the JSON exactly.

### `reports/responsible_ai_risk_report.md`

The integrated Responsible AI risk report. Static, reviewed documentation that cites the JSON reports above; it is not generated by a script.

---

## Limitations and Future Improvements

**Current limitations:**
- The baseline agent is fully deterministic. It does not learn or adapt across episodes.
- All randomness is in scenario sampling — the agent's policy is fixed.
- Scenarios are hand-crafted; a richer scenario set would provide a more challenging benchmark.
- Observation encoding is numeric (no text). This is intentional for simplicity but limits realism.
- The safety evaluation covers predefined cases for a deterministic, discrete-action agent — not an LLM. It does not test prompt injection, jailbreaks, or novel attack vectors, and passing it does not prove safety under all conditions.
- The guardrail is stateless, its fallback (ESCALATE) terminates the episode, and it is not exercised in the end-to-end baseline agent loop.
- The fairness evaluation uses synthetic groups (A/B) with 60 records each, measures outcome parity only, and includes no confidence intervals or significance tests. It is illustrative, not a real bias audit.
- No production monitoring, telemetry, or deployment exists; all evaluation is offline and deterministic.

**Future improvements:**
- Train a learned policy (e.g., PPO, DQN) using the same environment and compare against the baseline.
- Add more diverse scenarios and multi-field user responses.
- Extend the observation space with conversation history or a dialogue context vector.
- Integrate with SageMaker or Ray RLlib for scalable training experiments. (Not implemented: this project has no AWS or SageMaker integration.)
- Add a stochastic user model where responses vary based on frustration level.
- Add cross-turn guardrail state, confidence intervals for fairness metrics, and runtime monitoring before any production use.
