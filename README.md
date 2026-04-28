# Active Inference Agent - Implementation Summary

## Overview

This implementation provides a **production-ready Active Inference framework** for building autonomous agents that proactively minimize Expected Free Energy (EFE) before executing actions.

### Key Innovation

Unlike reactive agents that blindly follow instructions, this agent:
- **Predicts outcomes** before acting.
- **Calculates risk** using mathematical divergence metrics.
- **Quantifies uncertainty** via information theory.
- **Counter-Narrative Simulation (CNS)**: Actively hallucinates adversarial interpretations of its own actions to detect hidden security risks.
- **Parallel Testing Suite**: Observes internal states asynchronously via an event-driven control plane.

---

## Deliverables

### 1. **Parallel Testing Suite (Agent Control Plane)**
A decoupled safety layer that monitors the agent in real-time.
- **Event Bus (`events.py`)**: Asynchronous dispatcher for system-wide observability.
- **Execution Gate (`execution_gate.py`)**: Mandatory bottleneck for all tool actions. Passes actions through policies, CNS, and a Parallel Judge.
- **Outcome Simulator (`outcome_simulator.py`)**: Predicts consequences and runs adversarial simulations.
- **Expectation Checker (`expectation_checker.py`)**: Compares predicted vs. actual results via `step_id` correlation.

### 2. **free_energy.py**
Complete implementation of the Expected Free Energy calculation engine.
- **Risk (Pragmatic Value)**: Goal alignment via KL-Divergence.
- **Ambiguity (Epistemic Value)**: Uncertainty quantification via Shannon Entropy.

### 3. **agent_manager.py**
Main orchestrator implementing the cybernetic loop:
- **Planning**: DAG-based task decomposition.
- **Evaluation**: EFE-based policy selection and refinement.
- **Execution**: Governed by the Parallel Testing Suite.
- **Learning**: Post-task evaluation via LLM-as-a-Judge.

---

## Quick Start

### Running the Agent
```bash
python main.py "Research active inference and summarize key findings"
```

### Basic EFE Calculation
```python
from free_energy import ExpectedFreeEnergyEngine
efe_engine = ExpectedFreeEnergyEngine(efe_threshold=0.5)
result = efe_engine.compute_efe(policy, predictions, preferences)
```

---

## Mathematical Foundation

### Expected Free Energy
```
G(π) = Risk(π) + Ambiguity(π)
```
Where **Risk** measures goal divergence and **Ambiguity** measures state uncertainty.

---

## Key Features

- **Proactive Governance**: Actions are validated by the Parallel Testing Suite before execution.
- **Adversarial Audit**: CNS detects hidden malicious intent in benign-looking tasks.
- **Persistent Memory**: Vector-based semantic memory + Episodic logging.
- **HITL Controls**: Non-blocking human-in-the-loop approvals and asynchronous interrupts.

---

## Documentation

- [parallel_testing_walkthrough.md](../parallel_testing_walkthrough.md): Detailed guide to the safety suite and CNS.
- [enterprise_agent_feature_roadmap.md](../enterprise_agent_feature_roadmap.md): Current status of enterprise features.
- [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md): Deep dive into EFE mathematics.
