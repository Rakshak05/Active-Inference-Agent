# Active Inference Agent - Architecture Diagrams

## 1. Main Control Loop

```text
User Task
   |
   v
AgentManager
   |
   +--> Perception
   |      - encode goal
   |      - load memory context
   |
   +--> DAG Planning
   |      - generate subtasks
   |
   +--> Active Inference Loop
          - choose next subtask
          - select next action
          - estimate EFE
          - pass action through execution gate
          - execute tool
          - update beliefs and memory
          - continue until complete or halted
```

## 2. Execution Gate

```text
Proposed Action
   |
   v
Execution Gate
   |
   +--> Policy Engine
   |      - constitution checks
   |      - destructive action checks
   |
   +--> Low-Risk Fast Path?
   |      - yes -> narrative pressure test only
   |      - no  -> full parallel safety evaluation
   |
   +--> Parallel Safety Evaluation
   |      - outcome prediction
   |      - adversarial intent prediction
   |      - narrative pressure test
   |      - pragmatic judge
   |
   +--> Decision
          - allow
          - block
```

## 3. Parallel Testing Suite

```text
                         Event Bus
                            |
   -------------------------------------------------
   |                 |             |               |
   v                 v             v               v
Execution Gate  Expectation     Watchdog      OversightMemory
   |              Checker
   |
   +--> Outcome Simulator
   |      - predict outcome
   |      - predict adversarial intent
   |
   +--> Narrative Pressure Test
   |
   +--> Parallel Judge
```

## 4. Evidence-Grounded Research Flow

```text
Research Task
   |
   v
Search Memory ----> Memory Hits?
   |                    |
   |                    v
   |                relevant?
   |
   v
Web Search -------> Search Results?
   |                    |
   |                    v
   |                relevant?
   |
   v
Summarize
   |
   +--> if relevant evidence exists -> summary
   |
   +--> if evidence missing/irrelevant -> honest fallback
```

## 5. Performance-Oriented Paths

```text
Subtask
   |
   v
Heuristic Router
   |
   +--> common low-risk step?
   |      - search_memory
   |      - web_search
   |      - extract_info
   |      - report_answer
   |
   +--> yes -> skip planner LLM call
   +--> no  -> normal planner path
```

## 6. Failure Handling

```text
Tool Failure
   |
   v
Failure Feedback
   |
   +--> retry with replanning
   |
   +--> repeated low-information failure?
           - yes -> fail fast
           - no  -> continue loop
```

## 7. Research Safety Demonstrations

```text
Safety Benchmarks
   |
   +--> Narrative Pressure Invariance
   +--> Trojan Intent Reinterpretation
   +--> Counterfactual Evidence Deprivation
```
