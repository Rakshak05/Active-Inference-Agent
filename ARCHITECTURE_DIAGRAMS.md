# Active Inference Agent - Architecture Diagrams

## System Flow Diagram

```
                    USER TASK
                        │
                        ▼
        ┌───────────────────────────────┐
        │      AGENT MANAGER            │
        │   (Cybernetic Orchestrator)   │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   PHASE 1: PERCEPTION         │
        │   • Update beliefs            │
        │   • Encode goals              │
        └───────────────────────────────┘
                        │
                        ▼
        ╔═══════════════════════════════╗
        ║   PLANNING LOOP (Max 3x)      ║
        ╚═══════════════════════════════╝
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌──────────────┐                ┌──────────────┐
│ PHASE 2:     │                │ Generative   │
│ PLANNING     │◄───────────────│ Model        │
│              │                │              │
│ LLM          │                │ • Context    │
│ Interpreter  │                │ • Goals      │
└──────────────┘                └──────────────┘
        │
        ▼
┌──────────────────────┐
│ Generated Policy:    │
│ [action1, action2,   │
│  action3, ...]       │
└──────────────────────┘
        │
        ▼
┌──────────────┐
│ PHASE 3:     │
│ SIMULATION   │
│              │
│ Look-Ahead   │
│ Simulator    │
└──────────────┘
        │
        ▼
┌──────────────────────┐
│ Predicted Outcomes:  │
│ [outcome1, outcome2, │
│  outcome3, ...]      │
└──────────────────────┘
        │
        ▼
╔══════════════════════╗
║ PHASE 4: EVALUATION  ║
║                      ║
║ Free Energy Engine   ║
╚══════════════════════╝
        │
        ├─────────────────┐
        ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ Calculate    │  │ Calculate    │
│ RISK         │  │ AMBIGUITY    │
│              │  │              │
│ • KL-Div     │  │ • Entropy    │
│ • Goal       │  │ • State      │
│   Divergence │  │   Uncertainty│
│ • Constraint │  │ • Obs        │
│   Violation  │  │   Entropy    │
└──────────────┘  └──────────────┘
        │                 │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │   Total EFE     │
        │   G(π) = R + A  │
        └─────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ G(π) < θ ?      │
        └─────────────────┘
         │              │
    YES  │              │  NO
         ▼              ▼
    ┌────────┐    ┌──────────┐
    │ ACCEPT │    │ REJECT   │
    │        │    │          │
    │ Exit   │    │ Refine   │
    │ Loop   │    │ Prompt   │
    └────────┘    └──────────┘
         │              │
         │              └──────┐
         │                     │
         │              Back to PHASE 2
         │              (If attempts < max)
         │
         ▼
╔══════════════════════╗
║ PHASE 6: EXECUTION   ║
║                      ║
║ Toolgate (MCP)       ║
╚══════════════════════╝
         │
         ▼
┌──────────────────────┐
│ Actual Results       │
└──────────────────────┘
         │
         ▼
╔══════════════════════╗
║ PHASE 7: LEARNING    ║
║                      ║
║ Update World Model   ║
╚══════════════════════╝
         │
         ▼
    ┌────────┐
    │  DONE  │
    └────────┘
```

## EFE Calculation Pipeline

```
                POLICY + PREDICTIONS
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Expected Free Energy Engine │
        └───────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌─────────────────┐           ┌─────────────────┐
│  RISK Branch    │           │ AMBIGUITY Branch│
└─────────────────┘           └─────────────────┘
        │                               │
        ├─────────┬─────────┐          ├─────────┬─────────┐
        ▼         ▼         ▼          ▼         ▼         ▼
    ┌────┐   ┌────┐   ┌────┐      ┌────┐   ┌────┐   ┌────┐
    │Goal│   │Cons│   │Pred│      │State│  │Obs │   │Epis│
    │Div.│   │Vio.│   │Obs.│      │Unc. │  │Ent.│   │Acts│
    └────┘   └────┘   └────┘      └────┘   └────┘   └────┘
        │         │         │          │         │         │
        │         │         │          │         │         │
        │     ┌───▼─────────▼──┐       │      ┌──▼─────────▼──┐
        │     │ Vectorize Obs  │       │      │ Calculate     │
        │     │ Extract Features│       │      │ Entropy       │
        │     └────────┬────────┘       │      └───────┬───────┘
        │              │                │              │
        ▼              ▼                ▼              ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │ D_KL[Pred || Pref]   │      │ H(X) = -Σ P log P    │
    │                      │      │                      │
    │ KL-Divergence        │      │ Shannon Entropy      │
    └──────────────────────┘      └──────────────────────┘
                │                              │
                ▼                              ▼
        ┌───────────────┐            ┌───────────────┐
        │ w₁ × GoalDiv  │            │ w₃ × StateUnc │
        │ w₂ × ConsVio  │            │ w₄ × ObsEnt   │
        └───────────────┘            └───────────────┘
                │                              │
                └──────────┬───────────────────┘
                           ▼
                  ┌─────────────────┐
                  │ G(π) = R + A    │
                  │                 │
                  │ Total EFE       │
                  └─────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ EFEBreakdown    │
                  │                 │
                  │ • total_efe     │
                  │ • risk          │
                  │ • ambiguity     │
                  │ • components    │
                  │ • is_acceptable │
                  └─────────────────┘
```

## Mathematical Formulation

```
┌─────────────────────────────────────────────────────┐
│             EXPECTED FREE ENERGY                    │
│                                                     │
│  G(π) = E_Q(o|π)[D_KL[Q(s|o,π) || P(s|C)]         │
│                  - ln P(o|C)]                       │
│                                                     │
│       = Risk(π) + Ambiguity(π)                      │
└─────────────────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│   RISK (R)       │          │ AMBIGUITY (A)    │
│                  │          │                  │
│ Pragmatic Value  │          │ Epistemic Value  │
│ (Goal alignment) │          │ (Uncertainty)    │
└──────────────────┘          └──────────────────┘
         │                               │
         ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│ R = α × GD       │          │ A = γ × SU       │
│     β × CV       │          │     δ × OE       │
│                  │          │                  │
│ α = 0.7 (GoalDiv)│          │ γ = 0.6 (StateU) │
│ β = 0.3 (ConsVio)│          │ δ = 0.4 (ObsEnt) │
└──────────────────┘          └──────────────────┘
         │                               │
         ▼                               ▼
┌──────────────────┐          ┌──────────────────┐
│ Goal Divergence: │          │ State Uncertain: │
│                  │          │                  │
│ D_KL = Σ P log   │          │ U = 0.5 - 0.15×E │
│      (P(x)/Q(x)) │          │                  │
│                  │          │ E = epistemic    │
│ Measures how     │          │     actions      │
│ predicted ≠      │          │                  │
│ preferred        │          │ Reduced by       │
│                  │          │ search, query    │
└──────────────────┘          └──────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │ Observation Ent: │
                              │                  │
                              │ H = -Σ P log P   │
                              │                  │
                              │ Measures spread  │
                              │ of outcomes      │
                              └──────────────────┘
```

## Decision Flow

```
                    START
                      │
                      ▼
              ┌───────────────┐
              │    Generate   │
              │    Policy π   │
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │    Simulate   │
              │    Outcomes   │
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │   Calculate   │
              │      G(π)     │
              └───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │   G(π) < θ ?  │
              └───────────────┘
                 │         │
            YES  │         │  NO
                 ▼         ▼
          ┌──────────┐  ┌──────────┐
          │ EXECUTE  │  │ Attempts │
          │          │  │ < Max?   │
          └──────────┘  └──────────┘
                             │    │
                        YES  │    │  NO
                             ▼    ▼
                        ┌──────────────┐
                        │ Refine with  │
                        │ EFE feedback │
                        └──────────────┘
                             │         │
                             └─────┐   │
                                   │   │
                        Back to    │   │
                        Generate   │   │
                        Policy     │   │
                                   │   │
                                   │   ▼
                                   │  FAIL
                                   │
                                   ▼
                                 SUCCESS
```

## Component Interaction

```
┌─────────────────────────────────────────────────┐
│              GENERATIVE MODEL                   │
│                                                 │
│  ┌─────────────┐         ┌─────────────┐        │
│  │ Current     │         │ Preferred   │        │
│  │ Beliefs     │         │ Distribution│        │
│  │ Q(s)        │         │ P(s|C)      │        │
│  └─────────────┘         └─────────────┘        │
│         │                        │              │
│         └────────┬───────────────┘              │
└──────────────────┼──────────────────────────────┘
                   ▼
         ┌─────────────────┐
         │   CONTEXT       │
         └─────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ LLM INTERPRETER │
         └─────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │   POLICY π      │
         │                 │
         │ [a₁, a₂, ..., aₙ]│
         └─────────────────┘
                   │
                   ├─────────────────┐
                   ▼                 ▼
         ┌─────────────────┐  ┌─────────────────┐
         │ LOOK-AHEAD      │  │ FREE ENERGY     │
         │ SIMULATOR       │  │ ENGINE          │
         └─────────────────┘  └─────────────────┘
                   │                 │
                   ▼                 │
         ┌─────────────────┐        │
         │ PREDICTIONS     │        │
         │ [o₁, o₂, ..., oₙ]│────────┘
         └─────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ EFE BREAKDOWN   │
         │                 │
         │ Accept/Reject   │
         └─────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   ┌────────┐          ┌──────────┐
   │EXECUTE │          │ REFINE   │
   │        │          │          │
   │Toolgate│          │Re-prompt │
   └────────┘          └──────────┘
        │                     │
        ▼                     │
   ┌────────┐                │
   │UPDATE  │                │
   │BELIEFS │                │
   └────────┘                │
                             │
                             └──► Loop back
```

These diagrams illustrate:
1. **System Flow**: Complete cybernetic loop from perception to learning
2. **EFE Pipeline**: Detailed calculation breakdown
3. **Mathematical Structure**: Formal relationships between components
4. **Decision Logic**: Accept/reject/refine flow
5. **Component Interaction**: How modules communicate
