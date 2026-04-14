import json
from llm_gateway import LLMGateway

class LookAheadSimulator:
    """
    Simulates the outcome of proposed actions (Rollout module).
    Predicts the future observations resulting from a policy.
    """
    def __init__(self):
        self.gateway = LLMGateway()

    def simulate_policy(self, policy: list, current_context: dict) -> list:
        """
        Takes a sequence of steps (policy) and predicts the observations.
        Returns a 'Predicted Distribution' of outcomes.
        """
        predicted_observations = []
        simulated_context = current_context.copy()
        
        for step in policy:
            sys_prompt = (
                "You are an Environment Simulator. "
                "Given a context and an action, predict the result. "
                "Respond ONLY with a raw JSON object (no markdown) with exactly these keys:\n"
                "  \"predicted_outcome\": string describing what happens,\n"
                "  \"success_probability\": float 0-1 (how likely it succeeds),\n"
                "  \"risk_level\": float 0-1 (0=safe, 1=dangerous)\n"
                "Example: {\"predicted_outcome\": \"Folder deleted.\", \"success_probability\": 0.95, \"risk_level\": 0.1}"
            )
            user_prompt = (
                f"Current Context: {json.dumps(simulated_context)}\n"
                f"Action to take: {json.dumps(step)}\n"
                "Prediction:"
            )
            
            raw = self.gateway.generate_completion(sys_prompt, user_prompt, json_mode=True)
            
            # Try to parse structured response from LLM
            pred_obj = self._parse_prediction(raw, step)
            predicted_observations.append(pred_obj)
            
            # Update simulated context incrementally
            simulated_context[f"simulated_result_{len(predicted_observations)}"] = pred_obj["predicted_outcome"]
            
        return predicted_observations

    def _parse_prediction(self, raw: str, step: dict) -> dict:
        """
        Parse LLM prediction response. Falls back to heuristic if JSON fails.
        """
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "predicted_outcome" in parsed:
                # Clamp floats to valid range
                return {
                    "step": step,
                    "predicted_outcome": str(parsed.get("predicted_outcome", "")),
                    "success_probability": float(max(0.0, min(1.0, parsed.get("success_probability", 0.7)))),
                    "risk_level": float(max(0.0, min(1.0, parsed.get("risk_level", 0.3)))),
                    "tool": step.get("tool", "unknown")
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Heuristic fallback: infer from text content
        text = raw.strip() if isinstance(raw, str) else ""
        text_lower = text.lower()

        risk = 0.15  # default low risk
        success = 0.80

        danger_words = ["fail", "error", "danger", "critical", "cannot", "unable", "denied"]
        safe_words = ["success", "complete", "deleted", "removed", "done", "ok", "created"]

        if any(w in text_lower for w in danger_words):
            risk = 0.6
            success = 0.4
        elif any(w in text_lower for w in safe_words):
            risk = 0.1
            success = 0.92

        return {
            "step": step,
            "predicted_outcome": text or "Action executed.",
            "success_probability": success,
            "risk_level": risk,
            "tool": step.get("tool", "unknown")
        }

