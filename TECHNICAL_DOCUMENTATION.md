# Active Inference Agent - Technical Documentation

## Executive Summary

This project is an autonomous agent framework that combines:

- Active Inference inspired action evaluation
- tool-based execution
- a parallel safety control plane
- persistent memory
- evidence-aware fallback behavior

The design goal is not only to complete tasks, but to do so with measurable safety and better runtime control than a naive reactive agent.

## 1. Core Architecture

### 1.1 Main Components

- `agent_manager.py`
  - orchestrates the full control loop
  - tracks cycles, failures, execution history, and working context
- `llm_interpreter.py`
  - generates DAG plans and tool policies
- `look_ahead.py`
  - predicts policy outcomes for non-safe actions
- `free_energy.py`
  - computes Expected Free Energy
- `toolgate.py`
  - executes validated tools
- `memory/memory_manager.py`
  - semantic memory, episodic memory, and working context pruning
- `parallel_testing/*`
  - safety control plane around execution

### 1.2 Control Loop

The runtime loop is:

1. encode user goal
2. generate DAG-style subtasks
3. select next subtask
4. choose next action
5. estimate EFE
6. pass action through execution gate
7. execute the tool
8. update beliefs and memory
9. repeat until completion or halt

## 2. Expected Free Energy

The action selection mechanism is based on:

```text
G(pi) = Risk(pi) + Ambiguity(pi)
```

Where:

- **Risk** measures goal divergence and constraint violation
- **Ambiguity** measures uncertainty about predicted outcomes

The agent accepts low-EFE actions and refines or replans when EFE is too high.

## 3. Parallel Testing Suite

The parallel testing suite is the most important runtime safety layer.

### 3.1 Event Bus

`parallel_testing/events.py` provides a singleton asynchronous event bus.

It emits and distributes events such as:

- `TASK_STARTED`
- `PLAN_CREATED`
- `GATE_VALIDATION_STARTED`
- `OUTCOME_PREDICTED`
- `STEP_COMPLETED`
- `GATE_REJECTED`
- `TASK_COMPLETED`

This lets safety modules observe execution without being tightly coupled to the main agent loop.

### 3.2 Execution Gate

`parallel_testing/execution_gate.py` is the bottleneck for all tool actions.

Its behavior is:

1. run policy checks
2. if tool is low-risk, use fast-path validation
3. otherwise run higher-cost safety checks in parallel
4. allow or block the action

### 3.3 Policy Layer

`parallel_testing/policy_engine.py` enforces hard policy rules before LLM reasoning is used.

Examples:

- constitution-level blocks
- deletion without approval
- path protections

### 3.4 Outcome Simulation

`parallel_testing/outcome_simulator.py` predicts:

- likely result of an action
- side effects
- reversibility
- hidden adversarial interpretation

It includes Counter-Narrative Simulation, where the system asks how the same action could be maliciously reinterpreted.

### 3.5 Narrative Pressure Test

`parallel_testing/narrative_pressure_test.py` adds a custom safety simulation.

It checks whether the same action becomes unsafe when wrapped in:

- urgency
- authority
- secrecy
- euphemism
- false benevolence

This creates a more original benchmark than ordinary refusal-only testing.

### 3.6 Parallel Judge

`parallel_testing/parallel_judge.py` combines:

- predicted outcome
- CNS warning
- narrative pressure score

into a final pre-execution allow/block decision for higher-risk actions.

### 3.7 Expectation Checker

`parallel_testing/expectation_checker.py` compares predicted and actual outcomes.

If divergence is high, it emits `ANOMALY_DETECTED`.

### 3.8 Watchdog and Oversight Memory

- `watchdog.py` monitors repeated anomalies and rejection patterns
- `oversight_memory.py` stores gate rejections for later auditing

## 4. Runtime Optimizations

Several optimizations were added to make the project usable interactively.

### 4.1 Low-Risk Fast Path

Low-risk tools no longer pay the full simulation and judge cost.

Examples:

- `search_memory`
- `store_memory`
- `read_file`
- `web_search`
- `extract_info`
- `report_answer`

For these tools, the execution gate only performs lightweight checks.

### 4.2 Heuristic Action Routing

`agent_manager.py` now contains a heuristic router for repetitive low-risk subtasks.

This avoids calling the planner LLM for obvious patterns like:

- search memory
- web search
- summarize
- final answer

### 4.3 Fail-Fast Search Handling

Repeated empty web searches no longer loop for long periods.

After repeated low-information failures, the agent marks the subtask as degraded and moves forward.

### 4.4 Optional Post-Run Judge

The post-run LLM judge is now configurable.

By default:

- `ENABLE_POST_RUN_JUDGE=False`
- `JUDGE_SAMPLES=1`

This keeps the deep evaluator available without forcing every run to pay for it.

### 4.5 Extract-Info Fast Path

`adapters/data_adapters.py` now uses a cheap path for structured memory-style documents.

If evidence is irrelevant, it returns `Not found` instead of fabricating a vague summary.

## 5. Evidence-Grounded Answering

One of the most important recent upgrades is evidence-grounded fallback.

### 5.1 Problem Solved

Previously, the agent could:

- fail search
- still produce a generic answer
- look successful even when it had not verified anything

### 5.2 New Behavior

For research-like tasks, the final answer is now built with evidence checks:

- if relevant evidence exists, the summary is used
- if evidence is missing or irrelevant, the agent returns a transparent fallback

This behavior is implemented through:

- `_has_relevant_evidence()`
- `_build_final_answer()`

inside `agent_manager.py`.

## 6. Memory System

The memory layer includes:

- semantic memory through ChromaDB
- episodic action logging through SQLite
- working context compression using an LLM when the token budget is exceeded

This lets the agent retain:

- reusable factual context
- recent execution history
- condensed traces across long runs

## 7. Safety Model Summary

The project now combines several safety layers:

1. hard rule blocking
2. action simulation
3. adversarial reinterpretation
4. rhetorical-pressure testing
5. pragmatic pre-execution judging
6. predicted-vs-actual anomaly detection
7. audit logging

That layered design is stronger than relying on a single model refusal step.

## 8. Current Limitations

- the expectation checker still uses lightweight lexical overlap instead of semantic similarity
- research answers are safer now, but retrieval quality still limits answer quality
- some tool selection remains heuristic and can still be improved
- the post-run judge is disabled by default for speed, so deep evaluation is not always present

## 9. Recommended Next Enhancements

- semantic relevance scoring for retrieved evidence
- source citation support in final research answers
- benchmark runner for mentor/demo scenarios
- explicit offline mode for retrieval failure
- richer anomaly classification

## 10. Recommended Benchmarks

The strongest demonstrations for this repo are:

- **Narrative Pressure Invariance**
- **Trojan Intent Reinterpretation**
- **Counterfactual Evidence Deprivation**

Together they test:

- behavioral safety
- semantic safety
- epistemic honesty

## 11. Useful Configuration

```bash
EFE_THRESHOLD=0.65
MAX_REPLANS=3
LLM_TIMEOUT=90
ENABLE_POST_RUN_JUDGE=False
JUDGE_SAMPLES=1
DEBUG_MODE=True
```

## 12. Verification

Focused safety/runtime tests:

```bash
python -m unittest test_parallel_testing.py
```

## 13. Version Notes

This documentation reflects the current codebase state after:

- parallel testing suite integration
- narrative pressure test addition
- runtime fast-path optimization
- evidence-grounded fallback update
