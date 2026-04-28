import json
import os
from typing import Dict, Any, List
from .events import event_bus, AgentEvent

class OversightMemory:
    def __init__(self, storage_path: str = "data/oversight_memory.json"):
        self.storage_path = storage_path
        self.memory = self._load()
        event_bus.subscribe("GATE_REJECTED", self.handle_rejection)
        event_bus.subscribe("TASK_COMPLETED", self.handle_completion)

    def _load(self) -> Dict:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except:
                return {"rejections": [], "sessions": []}
        return {"rejections": [], "sessions": []}

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(self.memory, f, indent=2)

    async def handle_rejection(self, event: AgentEvent):
        self.memory["rejections"].append({
            "timestamp": event.timestamp,
            "action": event.payload.get("action"),
            "reason": event.payload.get("reason"),
            "source": event.payload.get("source")
        })
        self._save()

    async def handle_completion(self, event: AgentEvent):
        # Could store session summaries here
        pass

# Global instance
oversight_memory = OversightMemory()
