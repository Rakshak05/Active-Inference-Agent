import asyncio
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from .events import event_bus, AgentEvent

@dataclass
class Intent:
    objective: str
    risks: List[str]
    required_permissions: List[str]

class PlanInspector:
    def __init__(self):
        self.current_intent: Intent = None
        event_bus.subscribe("PLAN_CREATED", self.handle_plan_created)

    async def handle_plan_created(self, event: AgentEvent):
        goal = event.payload.get("goal", "")
        plan = event.payload.get("plan", [])
        
        # Simple extraction logic (could be LLM powered later)
        # For now, let's just identify some risks based on tools used in the plan
        risks = []
        permissions = ["safe"]
        
        plan_str = str(plan).lower()
        if "delete" in plan_str:
            risks.append("destructive_file_operation")
            permissions.append("restricted")
        if "http" in plan_str or "web_search" in plan_str:
            risks.append("network_access")
            permissions.append("internet")
        if "execute_python" in plan_str:
            risks.append("arbitrary_code_execution")
            permissions.append("restricted")
            
        self.current_intent = Intent(
            objective=goal,
            risks=risks,
            required_permissions=list(set(permissions))
        )
        print(f"[PlanInspector] Intent extracted: {self.current_intent}")

    def get_current_intent(self) -> Intent:
        return self.current_intent

# Global instance
plan_inspector = PlanInspector()
