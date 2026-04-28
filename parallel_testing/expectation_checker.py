import asyncio
import json
from typing import Dict, Any
from .events import event_bus, AgentEvent, emit_event

class ExpectationChecker:
    def __init__(self):
        self.pending_expectations = {}
        event_bus.subscribe("OUTCOME_PREDICTED", self.handle_prediction)
        event_bus.subscribe("STEP_COMPLETED", self.handle_completion)

    async def handle_prediction(self, event: AgentEvent):
        step_id = event.payload.get("step_id", "default")
        self.pending_expectations[step_id] = event.payload.get("prediction")
        # print(f"[ExpectationChecker] Prediction stored for {step_id}")

    async def handle_completion(self, event: AgentEvent):
        step_id = event.payload.get("step_id", "default")
        result = event.payload.get("result", {})
        actual_outcome = self._extract_actual_outcome(result)
        
        if step_id in self.pending_expectations:
            expected = self.pending_expectations.pop(step_id)
            divergence = self.calculate_divergence(expected, actual_outcome)
            
            if divergence > 0.7:
                await emit_event("ANOMALY_DETECTED", step_id=step_id, divergence=divergence, expected=expected, actual=actual_outcome)
                # print(f"[ExpectationChecker] ALERT: High divergence detected for {step_id}!")

    def calculate_divergence(self, expected: Dict, actual: str) -> float:
        """
        Calculates divergence between expected outcome dict and actual outcome string.
        0.0 means perfect match, 1.0 means total mismatch.
        """
        expected_text = str(expected.get("predicted_outcome", "")).lower()
        actual_text = str(actual).lower()
        
        if not expected_text or not actual_text:
            return 0.5
            
        # Very simple keyword-based overlap as a placeholder for LLM comparison
        exp_words = set(expected_text.split())
        act_words = set(actual_text.split())
        
        if not exp_words:
            return 0.5
            
        overlap = len(exp_words.intersection(act_words))
        similarity = overlap / len(exp_words)
        
        return 1.0 - similarity

    def _extract_actual_outcome(self, result: Dict[str, Any]) -> str:
        """
        Normalize AgentManager execution records into a single comparable string.
        """
        if isinstance(result, dict):
            if result.get("actual_outcome"):
                return str(result["actual_outcome"])

            tool_results = result.get("results", [])
            if isinstance(tool_results, list):
                parts = []
                for entry in tool_results:
                    if entry.get("actual_outcome"):
                        parts.append(str(entry["actual_outcome"]))
                    elif entry.get("error"):
                        parts.append(str(entry["error"]))
                if parts:
                    return " | ".join(parts)

        return ""

# Global instance
expectation_checker = ExpectationChecker()
