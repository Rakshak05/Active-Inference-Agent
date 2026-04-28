# Active Inference Agent

## Overview

This project implements an Active Inference inspired autonomous agent with a safety-oriented control plane.

The agent does not execute tool actions directly from a single planner output. Instead, it:

- decomposes tasks into subtasks
- simulates and evaluates actions before execution
- applies a parallel testing suite as an execution gate
- records outcomes and anomalies for later review
- uses evidence-grounded fallback behavior for research-style tasks

## Current Highlights

- **Active Inference loop**: planning, simulation, evaluation, execution, and learning
- **Parallel Testing Suite**: policy gate, outcome simulation, CNS, narrative pressure testing, watchdogs, and oversight memory
- **Fast-path optimizations**: low-risk tools skip expensive safety/judge overhead
- **Fail-fast behavior**: repeated dead-end searches no longer loop indefinitely
- **Evidence-grounded answering**: research tasks avoid confident summaries when external evidence is missing
- **Persistent memory**: semantic memory with episodic logging

## Key Safety Features

### Counter-Narrative Simulation

The agent simulates a malicious reinterpretation of a proposed action before execution. This helps detect dual-use or Trojan-horse prompts where a harmful task is disguised as a helpful one.

### Narrative Pressure Test

The agent also tests whether a proposed action becomes unsafe under rhetorical pressure such as:

- authority
- urgency
- secrecy
- euphemism
- false benevolence

This is implemented in `parallel_testing/narrative_pressure_test.py`.

### Execution Gate

All actions pass through `parallel_testing/execution_gate.py`.

For low-risk tools, the gate uses a lightweight fast path.

For higher-risk tools, the gate runs:

- outcome prediction
- adversarial reinterpretation
- narrative pressure scoring
- pragmatic judging

## Runtime Behavior

The current runtime includes several optimizations:

- heuristic routing for repetitive low-risk subtasks
- optional post-run LLM judge
- single-sample judge by default
- lower LLM timeout defaults
- `extract_info` fast path for structured memory-style inputs

These changes improve responsiveness without removing the core safety gate.

## Evidence-Grounded Fallback

For research-like tasks, the agent now checks whether retrieved evidence is actually relevant before presenting a summary.

If retrieval fails or evidence is irrelevant, the agent uses a transparent fallback instead of pretending it verified the topic.

## Main Files

- `agent_manager.py`: main orchestrator and active inference control loop
- `free_energy.py`: EFE computation engine
- `llm_interpreter.py`: policy and DAG planning
- `llm_gateway.py`: model gateway for local or remote LLM calls
- `toolgate.py`: tool execution layer
- `parallel_testing/events.py`: event bus
- `parallel_testing/execution_gate.py`: action validation gate
- `parallel_testing/outcome_simulator.py`: outcome and adversarial simulation
- `parallel_testing/parallel_judge.py`: pre-execution judging
- `parallel_testing/expectation_checker.py`: predicted vs actual divergence tracking
- `parallel_testing/watchdog.py`: anomaly watcher
- `parallel_testing/oversight_memory.py`: rejection and oversight audit log
- `parallel_testing/narrative_pressure_test.py`: rhetorical manipulation safety simulation

## Running the Agent

```bash
python main.py "Research active inference and summarize key findings"
```

## Useful Configuration

Environment variables:

```bash
EFE_THRESHOLD=0.65
MAX_REPLANS=3
LLM_TIMEOUT=90
ENABLE_POST_RUN_JUDGE=False
JUDGE_SAMPLES=1
DEBUG_MODE=True
```

## Testing

Focused safety and runtime checks:

```bash
python -m unittest test_parallel_testing.py
```

## Documentation

- `TECHNICAL_DOCUMENTATION.md`: technical system description
- `ARCHITECTURE_DIAGRAMS.md`: high-level architecture and control-plane diagrams

## Current Limitations

- research-quality answers still depend on retrieval quality
- the expectation checker uses lightweight text overlap instead of semantic comparison
- the post-run judge is now optional for speed, so deep post-hoc evaluation may be skipped in fast mode

## Recommended Demo Scenarios

- Narrative Pressure Invariance
- Trojan Intent Reinterpretation
- Counterfactual Evidence Deprivation

