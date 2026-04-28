import asyncio
from typing import Dict, Any, List, Optional
from .events import emit_event

from .policy_engine import policy_engine
from .parallel_judge import parallel_judge
from .outcome_simulator import outcome_simulator
from .narrative_pressure_test import narrative_pressure_test

class ExecutionGate:
    LOW_RISK_TOOLS = {
        "search_memory",
        "store_memory",
        "read_file",
        "list_directory",
        "check_path",
        "read_json",
        "read_csv",
        "filter_records",
        "transform_records",
        "slice_records",
        "get_field_values",
        "extract_info",
        "evaluate",
        "web_search",
        "http_get",
        "log_message",
        "report_answer",
    }

    def __init__(self):
        self.policies = [policy_engine.evaluate]
        self.llm_judge = parallel_judge

    def register_policy(self, policy_fn):
        self.policies.append(policy_fn)

    async def validate_action(self, action: List[Dict], context: Dict, step_id: str = "default") -> Dict[str, Any]:
        """
        Validates an action against policies, outcome simulation, and LLM judge.
        """
        await emit_event("GATE_VALIDATION_STARTED", action=action, step_id=step_id)

        # 1. Check Policies
        for policy in self.policies:
            is_allowed, reason = policy(action, context)
            if not is_allowed:
                await emit_event("GATE_REJECTED", action=action, reason=reason, source="policy", step_id=step_id)
                return {"allowed": False, "reason": f"Policy Block: {reason}"}

        step = action[0] if action else {}
        tool_name = step.get("tool", "")

        # Fast path: low-risk tools do not need 2-3 extra LLM calls per step.
        if tool_name in self.LOW_RISK_TOOLS:
            pressure_test = await narrative_pressure_test.evaluate(action, context)
            context["narrative_pressure_test"] = pressure_test
            if pressure_test.get("pressure_score", 0.0) >= 0.65:
                reason = pressure_test.get("summary", "Narrative pressure test rejected the action.")
                await emit_event("GATE_REJECTED", action=action, reason=reason, source="narrative_pressure", step_id=step_id)
                return {"allowed": False, "reason": f"Narrative Pressure Reject: {reason}"}

            await emit_event("GATE_APPROVED", action=action, step_id=step_id)
            return {"allowed": True, "reason": "Low-risk fast path approved"}

        # 2. Predict Outcome, Adversarial Intent, and Narrative Pressure in parallel
        prediction, adversarial_check, pressure_test = await asyncio.gather(
            outcome_simulator.predict_outcome(step, context, step_id=step_id),
            outcome_simulator.predict_adversarial_intent(step, context),
            narrative_pressure_test.evaluate(action, context),
        )
        
        context["latest_prediction"] = prediction
        context["adversarial_check"] = adversarial_check
        context["narrative_pressure_test"] = pressure_test

        if pressure_test.get("pressure_score", 0.0) >= 0.65:
            reason = pressure_test.get("summary", "Narrative pressure test rejected the action.")
            await emit_event("GATE_REJECTED", action=action, reason=reason, source="narrative_pressure", step_id=step_id)
            return {"allowed": False, "reason": f"Narrative Pressure Reject: {reason}"}
        
        # 3. LLM Judge evaluation
        if self.llm_judge:
            is_allowed, reason = await self.llm_judge.evaluate_action(
                task=context.get("task_instruction", ""),
                action=action,
                context=context
            )
            
            # If adversarial plausibility is extremely high, we might override and block
            if adversarial_check.get("plausibility_score", 0) > 0.95:
                is_allowed = False
                reason = f"CRITICAL CNS ALERT: Highly plausible attack narrative detected: {adversarial_check.get('attack_narrative')}"

            if not is_allowed:
                await emit_event("GATE_REJECTED", action=action, reason=reason, source="judge", step_id=step_id)
                return {"allowed": False, "reason": f"Judge Reject: {reason}"}
        
        await emit_event("GATE_APPROVED", action=action, step_id=step_id)
        return {"allowed": True, "reason": "All checks passed"}

execution_gate = ExecutionGate()
