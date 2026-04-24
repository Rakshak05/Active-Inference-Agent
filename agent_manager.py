"""
Enhanced Agent Manager - Active Inference Orchestrator
======================================================
Main control loop implementing the Active Inference cycle with
sophisticated re-planning and Free Energy minimization.

Cybernetic Loop:
1. Perception: Receive task and update beliefs
2. Planning: Generate policy via LLM
3. Simulation: Predict outcomes via look-ahead
4. Evaluation: Compute Expected Free Energy (EFE)
5. Inference: If EFE too high, refine and re-plan
6. Execution: Execute validated low-EFE policy
7. Learning: Update world model with actual outcomes
"""

import sys
import os
import json
import signal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

class HITLInterruptHandler:
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self.handle_interrupt)

    def handle_interrupt(self, sig, frame):
        if self.interrupted:
            print("\n[HITL] Force Quitting...")
            sys.exit(1)
        print("\n\n[HITL] Interrupt signal received! Pausing at the next safe checkpoint.")
        print("Press Ctrl+C again to force quit immediately.")
        self.interrupted = True

# Ensure the parent directory is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import framework components
from generative_model import GenerativeModel
from llm_interpreter import LLMInterpreter
from look_ahead import LookAheadSimulator
from toolgate import Toolgate
from adapters.filesystem_adapters import setup_filesystem_adapters
from adapters.communication_adapters import setup_communication_adapters
from adapters.data_adapters import setup_data_adapters
from adapters.web_adapters import setup_web_adapters
from adapters.code_adapters import setup_code_adapters
from config import config

# Import the enhanced free energy engine
from free_energy import ExpectedFreeEnergyEngine, EFEBreakdown
from memory.memory_manager import MemoryManager
from planning.planner import DAGTracker


@dataclass
class PlanningAttempt:
    """Record of a single planning attempt."""
    attempt_number: int
    timestamp: str
    policy: List[Dict]
    predictions: List[Dict]
    efe_breakdown: EFEBreakdown
    refinement_applied: Optional[str] = None


class AgentManager:
    """
    Main orchestrator of the Active Inference Agent.
    
    This class implements the complete cybernetic loop:
    - Maintains world model (beliefs + preferences)
    - Generates and evaluates policies
    - Minimizes Expected Free Energy through iterative refinement
    - Executes validated plans
    - Updates beliefs based on observations
    """
    
    def __init__(self, efe_threshold: Optional[float] = None, max_replans: Optional[int] = None, session_id: Optional[str] = None):
        """
        Initialize the Agent Manager.
        
        Args:
            efe_threshold: Override config EFE threshold
            max_replans: Override config max replanning attempts
            session_id: Multi-Session identifier UUID
        """
        import uuid
        self.session_id = session_id or str(uuid.uuid4())
        
        # Core components
        self.world_model = GenerativeModel()
        self.interpreter = LLMInterpreter()
        self.simulator = LookAheadSimulator()
        self.toolgate = Toolgate()
        
        # Initialize Persistent Memory System with isolated session logic
        self.memory = MemoryManager(session_id=self.session_id)
        
        # Knowledge Ingestion System
        from knowledge_ingestion import KnowledgeIngestor
        self.ingestor = KnowledgeIngestor(self.memory)
        
        # Start watching a specific workspace / docs directory (if configured)
        if getattr(config, 'WORKSPACE_DIR', None):
            self.ingestor.watch_directory(config.WORKSPACE_DIR)
        
        # Initialize EFE engine with configurable threshold
        threshold = efe_threshold if efe_threshold is not None else config.EFE_THRESHOLD
        self.efe_engine = ExpectedFreeEnergyEngine(efe_threshold=threshold)
        
        # Planning parameters
        self.max_replans = max_replans if max_replans is not None else config.MAX_REPLANS
        
        # History tracking
        self.planning_history: List[PlanningAttempt] = []
        self.execution_history: List[Dict] = []
        
        # Setup tool adapters
        setup_filesystem_adapters(self.toolgate)
        setup_communication_adapters(self.toolgate)
        setup_data_adapters(self.toolgate)
        setup_web_adapters(self.toolgate)
        setup_code_adapters(self.toolgate)
        
        # Native memory adapters
        self.toolgate.register_adapter("store_memory", lambda step: self.memory.store_semantic_knowledge(step.get("args", {}).get("fact", "empty"), step.get("args", {}).get("metadata", {})) or "Fact successfully committed to long-term memory.")
        self.toolgate.register_adapter("search_memory", lambda step: self.memory.retrieve_semantic_knowledge(step.get("args", {}).get("query", ""), n_results=5))
        
        print(f"[Agent Manager] Initialized with EFE threshold: {threshold}")
        print(f"[Agent Manager] Maximum replanning attempts: {self.max_replans}")
    
    def process_task(self, user_instruction: str, max_steps: int = 15) -> Dict:
        """
        Main entry point: Process a user task through the Active Inference loop.
        
        Args:
            user_instruction: Natural language task description
            max_steps: Maximum number of active inference cycles
            
        Returns:
            Dictionary containing execution results and metadata
        """
        print("\n" + "="*80)
        print(f"[Agent Manager] NEW TASK: {user_instruction}")
        print("="*80)
        
        # Reset history for this task
        self.planning_history = []
        self.execution_history = []
        self.current_task = user_instruction
        
        # Persistent variable store shared across ALL cycles (fixes cross-cycle $var loss)
        self._cycle_var_store: dict = {}
        
        # PHASE 1: PERCEPTION
        print("\n[PHASE 1: PERCEPTION]")
        self.world_model.set_preference(user_instruction)
        print(f"✓ Goal encoded in generative model")
        
        # PHASE 1.5: PLANNING
        print("\n[PHASE 1.5: PLANNING & DAG DECOMPOSITION]")
        context = self.world_model.get_context()
        context["semantic_memory"] = self.memory.retrieve_semantic_knowledge(user_instruction, n_results=3)
        
        raw_plan = self.interpreter.generate_dag_plan(user_instruction, context)
        self.dag = DAGTracker()
        self.dag.load_from_json(raw_plan)
        
        # Asynchronous Interrupts Handler
        hitl_interrupt = HITLInterruptHandler()
        
        # CONTINUOUS ACTIVE INFERENCE LOOP
        print("\n[PHASE 2: ACTIVE INFERENCE LOOP]")
        
        step = 0
        while step < max_steps and not self.dag.is_fully_completed():
            step += 1
            print(f"\n--- Cycle {step}/{max_steps} ---")
            
            # Asynchronous Interrupt Processing
            if hitl_interrupt.interrupted:
                print("\n[HITL Asynchronous Interrupt] Agent execution paused by User.")
                choice = input("Do you want to (r)esume, (m)odify plan, or (q)uit? [r/m/q]: ").strip().lower()
                hitl_interrupt.interrupted = False
                if choice == 'q':
                    print("User terminated the session via HITL.")
                    break
                elif choice == 'm':
                    print("Current DAG Plan:")
                    print(json.dumps([{"id": t.id, "desc": t.description} for t in self.dag.tasks.values() if t.status != "completed"], indent=2))
                    new_plan_str = input("Enter new overriding DAG JSON array (or press enter to cancel): ")
                    if new_plan_str:
                        try:
                            self.dag.load_from_json(json.loads(new_plan_str))
                            print("Plan overridden successfully.")
                            continue
                        except Exception as e:
                            print(f"Invalid JSON: {e}. Resuming with old plan.")
                            
            print(self.dag.get_plan_state())
            
            ready_tasks = self.dag.get_ready_tasks()
            if not ready_tasks:
                print("No tasks ready. Possible deadlock or failed subtask.")
                break
                
            current_subtask = ready_tasks[0]
            self.dag.start_task(current_subtask.id)
            print(f"> Executing Subtask [{current_subtask.id}]: {current_subtask.description}")
            
            # Build context: world model state + completed steps summary
            context = self.world_model.get_context()
            context["completed_steps"] = self._summarise_completed_steps()
            context["available_vars"] = list(self._cycle_var_store.keys())
            
            # Inject Persistent Memory
            context["semantic_memory"] = self.memory.retrieve_semantic_knowledge(current_subtask.description, n_results=3)
            context["recent_episodes"] = self.memory.get_recent_episodes(limit=3)
            
            policy, predictions, efe_breakdown = self._evaluate_next_action(current_subtask.description, context)
            
            if not policy:
                print("Unable to determine safe next action. Failing subtask.")
                self.dag.fail_task(current_subtask.id, error="Failed to generate safe policy.")
                break
                
            # Execute the NEXT action only
            next_action = [policy[0]]
            action_tool = next_action[0].get('tool')
            
            # Confidence Scoring
            if efe_breakdown:
                confidence = max(0.0, min(1.0, 1.0 - efe_breakdown.ambiguity))
                print(f"[Self-Evaluation] Confidence Score: {confidence*100:.1f}%")
            
            # Clarification Protocol - Bypass for native internal cognition tools or low risk actions
            is_cognitive_tool = action_tool in ("store_memory", "search_memory")
            has_low_risk = efe_breakdown and (efe_breakdown.risk < 0.2 and efe_breakdown.risk_components.get("constraint_violation", 0.0) == 0.0)

            if efe_breakdown and efe_breakdown.ambiguity > 0.6 and not (is_cognitive_tool or has_low_risk):
                print(f"\n[HITL Clarification Protocol] Agent confidence is low (Ambiguity: {efe_breakdown.ambiguity:.2f}).")
                print(f"Subtask: {current_subtask.description}")
                if policy:
                    print(f"Proposed policy: {json.dumps(policy, indent=2)}")
                clarification = input("Please provide clarification or guidance (or press enter to let agent try anyway): ").strip()
                if clarification:
                    print("[RE-PLANNING HOOK] Adjusting task description based on your clarification...")
                    self.dag.tasks[current_subtask.id].description += "\nUser Clarification: " + clarification
                    self.dag.tasks[current_subtask.id].status = "pending" # Reset status so we try again
                    continue
            
            # Pre-Action Critique
            HIGH_RISK_CRITIQUE_TOOLS = {"delete_file", "delete_folder", "create_directory", "execute_python", "http_post", "send_email", "send_emails_bulk"}
            if action_tool in HIGH_RISK_CRITIQUE_TOOLS:
                print("[Self-Evaluation] Running internal Pre-Action Critique on high-risk tool...")
                is_valid, critique_feedback = self.interpreter.critique_policy(next_action, context)
                if not is_valid:
                    print(f"  └─ [Critique Failed] {critique_feedback}")
                    print("  └─ [Fix-It Loop] Retrying action evaluation with safety feedback injected...")
                    context["failure_feedback"] = f"CRITIQUE REJECTED YOUR POLICY: {critique_feedback} - Fix the plan."
                    self.dag.tasks[current_subtask.id].status = "pending" # Retry
                    continue
                else:
                    print("  └─ [Critique Passed] Internal safety and logic verified.")
            
            # Explicit completion signal from LLM
            if action_tool in ("task_complete", "end_task", "done"):
                print("Agent signalled subtask complete.")
                self.dag.complete_task(current_subtask.id)
                continue

            execution_result = self._execute_step(next_action, efe_breakdown)
            
            # Update Beliefs with Observation
            self.world_model.update_beliefs(execution_result)
            
            if execution_result.get("status") == "success":
                print("[Self-Evaluation] Running Post-Action Validation...")
                
                # Strip out 'raw' data from results to prevent massive LLM contexts
                clean_execution = {**execution_result}
                clean_execution["results"] = [
                    {k: v for k, v in r.items() if k != "raw"} 
                    for r in clean_execution.get("results", [])
                ]
                
                is_successful, validation_feedback = self.interpreter.validate_outcome(
                    user_instruction, current_subtask.description, json.dumps(clean_execution)
                )
                
                if not is_successful:
                    print(f"  └─ [Validation Failed] {validation_feedback}")
                    print("  └─ [Fix-It Loop] Re-routing execution to fix the problem...")
                    execution_result["status"] = "error" # Override status to force failure track
                    execution_result["error"] = validation_feedback
                else:
                    print("  └─ [Validation Passed] Step outcome aligns with goal.")

            # Stop if there was a critical failure
            if execution_result.get("status") == "error":
                print(f"Subtask execution error: {execution_result}")
                
                # RE-PLANNING HOOK / Internal self-correction iteration
                print("[RE-PLANNING HOOK] Feeding error back as an observation for self-correction...")
                # Extract specific error for better feedback
                error_msg = execution_result["results"][-1].get("error", str(execution_result)) if execution_result.get("results") else str(execution_result)
                context["failure_feedback"] = f"Action {action_tool} failed: {error_msg}. Analyze the error and generate a new action to fix the issue."
                
                # Keep task pending to retry locally instead of failing the whole DAG
                self.dag.tasks[current_subtask.id].status = "pending"
                continue

            # ── Automatic task-completion detection ────────────────────────────
            # If a write/create operation just succeeded, check whether all
            # goal artefacts (files named in the instruction) now exist.
            if execution_result.get("status") == "success":
                if self._is_task_complete(current_subtask.description, action_tool):
                    print(f"[Agent Manager] Subtask '{current_subtask.id}' completion detected.")
                    self.dag.complete_task(current_subtask.id)
                
        # Generate final report
        task_done = self.dag.is_fully_completed()
        status = "success" if task_done else ("max_steps_reached" if step >= max_steps else "halted")
        final_report = {
            "status": status,
            "task": user_instruction,
            "cycles_completed": step,
            "execution_history": self.execution_history,
            "timestamp": datetime.now().isoformat()
        }
        
        print("\n" + "="*80)
        print("[TASK COMPLETE]")
        print("="*80)
        
        return final_report

    # ── helpers for context-aware looping ─────────────────────────────────────

    def _summarise_completed_steps(self) -> list:
        """Build a concise list of what's already been executed this task, bounded by token limits."""
        return self.memory.get_working_context()

    _WRITE_TOOLS = {"write_file", "write_csv", "write_json", "create_directory", "copy_file", "move_file", "download_file"}

    def _is_task_complete(self, instruction: str, last_tool: str) -> bool:
        """
        Heuristic: if the last action was a write/create tool and every filename
        mentioned in the instruction that looks like an output file now exists on disk.
        """
        if last_tool not in self._WRITE_TOOLS:
            return False
        import re, os
        # Extract quoted filenames or filenames ending in known extensions
        candidates = re.findall(
            r"['\"]([^'\"]+\.[a-z]{2,5})['\"]|\b([\w\-]+\.(?:txt|csv|json|md|log|py|js|html|xml|yaml))\b",
            instruction, re.IGNORECASE
        )
        output_files = [m[0] or m[1] for m in candidates]
        if not output_files:
            return False
        return all(os.path.exists(f) for f in output_files)
    
    def _evaluate_next_action(self, instruction: str, context: dict) -> Tuple[Optional[List[Dict]], Optional[List[Dict]], Optional[EFEBreakdown]]:
        """
        Evaluates EFE to select the best next immediate action.
        Passes completed-step history so the LLM knows what's done.
        """
        completed = context.get("completed_steps", [])
        avail_vars = context.get("available_vars", [])

        # Build an awareness prefix so the LLM doesn't repeat completed steps
        if completed:
            done_str = json.dumps(completed, indent=2)
            vars_str = ", ".join(f"${v}" for v in avail_vars) if avail_vars else "none"
            awareness = (
                f"\n\nSTEPS ALREADY COMPLETED THIS TASK (do NOT repeat these):\n{done_str}"
                f"\nVARIABLES IN MEMORY: {vars_str}"
                f"\n\nOutput ONLY the VERY NEXT single action still needed to finish the task."
                f" If the task is fully done, output: [{{\"tool\": \"task_complete\", \"args\":{{}}}}]"
            )
        else:
            awareness = (
                "\n\nNo steps have been completed yet. "
                "Output ONLY the VERY NEXT single action to begin the task."
            )

        if "failure_feedback" in context:
            awareness += f"\n\nRECENT FAILURE OBSERVATION:\n{context['failure_feedback']}\nAdjust your next action to fix this issue."
            # Clear feedback after consuming it
            del context["failure_feedback"]

        refined_instruction = instruction
        
        for attempt in range(1, self.max_replans + 1):
            policy = self.interpreter.generate_policy(
                refined_instruction + awareness,
                context
            )
            
            if not policy:
                continue

            # If LLM signals completion, relay it immediately
            if policy[0].get("tool") in ("task_complete", "end_task", "done"):
                return policy, [], None  # efe_breakdown not needed for completion
                
            predictions = self.simulator.simulate_policy(policy, context)
            preferences = self.world_model.preferences.get_preferences()
            efe_breakdown = self.efe_engine.compute_efe(policy, predictions, preferences, context=context)
            
            print(f"--- Agent Analysis (Attempt {attempt}) ---")
            print(efe_breakdown)
            
            # Record this attempt
            self.planning_history.append(PlanningAttempt(
                attempt_number=attempt,
                timestamp=datetime.now().isoformat(),
                policy=policy,
                predictions=predictions,
                efe_breakdown=efe_breakdown,
                refinement_applied=refined_instruction if attempt > 1 else None
            ))
            
            if efe_breakdown.is_acceptable:
                return policy, predictions, efe_breakdown
                
            # Auto-refine if unacceptable
            refinements = ["[REFINEMENT NEEDED] EFE Assessor rejected the previous proposed plan."]
            if efe_breakdown.ambiguity > 0.5:
                refinements.append("High Uncertainty detected. Do NOT take destructive actions yet. Output an information-gathering step FIRST (e.g., check_path, list_directory).")
            if efe_breakdown.risk > 0.5:
                refinements.append("High Risk detected. Ensure the action aligns precisely with the goal.")
            
            refined_instruction = f"{instruction}\n" + " ".join(refinements) + f" (Attempt {attempt+1})"
            
        return None, None, None

    def _execute_step(self, action: List[Dict], efe_breakdown: Optional[EFEBreakdown]) -> Dict:
        """
        Execute a single action via Toolgate, using the persistent cross-cycle
        variable store so $var references survive between cycles.
        """
        from security_constitution import check_policy_against_constitution
        violations = check_policy_against_constitution(action)
        if violations:
            raise Exception(f"SAFETY VIOLATION: {violations}")

        results = []
        for step in action:
            tool_name = step.get("tool", "")
            resolved_args = self.toolgate._resolve(step.get("args", {}), self._cycle_var_store)
            resolved_step = {**step, "args": resolved_args}

            HIGH_RISK_TOOLS = {"delete_file", "delete_folder", "create_directory", "execute_python", "http_post", "send_email", "send_emails_bulk"}
            
            if tool_name in HIGH_RISK_TOOLS:
                print(f"\n[HITL Checkpoint] The agent wants to execute a HIGH-RISK tool: '{tool_name}'")
                print(f"Arguments: {json.dumps(resolved_step.get('args', {}), indent=2)}")
                approved = False
                while not approved:
                    choice = input("[HITL] Do you approve? (y)es, (n)o, (e)dit: ").strip().lower()
                    if choice in ['y', 'yes', '']:
                        approved = True
                    elif choice in ['n', 'no']:
                        raise Exception("HITL Validation Failed: User rejected the pending action.")
                    elif choice in ['e', 'edit']:
                        new_args = input("Enter new overriding JSON arguments: ")
                        try:
                            resolved_step["args"] = json.loads(new_args)
                            print("Override applied successfully.")
                            # Still ask to approve just in case they typo'd
                        except json.JSONDecodeError:
                            print("Invalid JSON. Try again.")

            if tool_name in self.toolgate.adapters:
                print(f"[Toolgate]   Running '{tool_name}' …")
                try:
                    raw_result = self.toolgate.adapters[tool_name](resolved_step)
                    output_var = step.get("output_var")
                    if output_var:
                        self._cycle_var_store[output_var] = raw_result
                        print(f"[Toolgate]     └─ saved to ${output_var}")
                    results.append({
                        "step": step,
                        "actual_outcome": self.toolgate._summarise(raw_result),
                        "raw": raw_result,
                        "status": "success",
                    })
                except Exception as e:
                    print(f"[Toolgate] X '{tool_name}' raised: {e}")
                    results.append({"step": step, "error": str(e), "status": "error"})
            else:
                print(f"[Toolgate] No adapter for '{tool_name}'. Skipping.")
                results.append({
                    "step": step,
                    "actual_outcome": f"No adapter for '{tool_name}'",
                    "status": "skipped",
                })

        efe_val = efe_breakdown.total_efe if efe_breakdown else 0.0
        any_error = any(r.get("status") == "error" for r in results)
        
        # Log episode to persistent memory and working context
        for r in results:
            self.memory.log_episode(
                task=getattr(self, "current_task", "unknown_task"),
                tool=r["step"].get("tool", "unknown"),
                args=r["step"].get("args", {}),
                result=r.get("actual_outcome", r.get("error", "unknown")),
                efe_score=efe_val
            )
            # 3. Add to Working Context (pruned automatically by tokens)
            self.memory.add_working_context({
                "tool": r["step"].get("tool", "unknown"),
                "outcome": r.get("actual_outcome", r.get("error", "unknown"))
            })

        record = {
            "status": "error" if any_error else "success",
            "results": results,
            "efe": efe_val,
            "timestamp": datetime.now().isoformat()
        }
        self.execution_history.append(record)
        return record
            
    def get_planning_history(self) -> List[PlanningAttempt]:
        return self.planning_history
    
    def get_execution_history(self) -> List[Dict]:
        return self.execution_history
    
    def export_session_log(self, filepath: str = "session_log.json"):
        session_data = {
            "execution_history": self.execution_history,
            "timestamp": datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        print(f"Session log exported to {filepath}")


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("ACTIVE INFERENCE AGENT - ENHANCED MANAGER TEST")
    print("="*80)
    
    # Initialize agent with custom settings
    agent = AgentManager(efe_threshold=0.5, max_replans=3)
    
    # Test task 1: Information gathering (should have low ambiguity)
    print("\n\n" + "="*80)
    print("TEST 1: Information Gathering Task")
    print("="*80)
    
    result1 = agent.process_task(
        "Research the latest developments in active inference and summarize key findings"
    )
    
    print(f"\nResult: {json.dumps(result1, indent=2)}")
    
    # Test task 2: High-risk action (should trigger re-planning)
    print("\n\n" + "="*80)
    print("TEST 2: High-Risk Action Task")
    print("="*80)
    
    result2 = agent.process_task(
        "Delete all files in the system directory"
    )
    
    print(f"\nResult: {json.dumps(result2, indent=2)}")
    
    # Export session log
    agent.export_session_log("enhanced_agent_test_log.json")
