import asyncio
from .events import event_bus, AgentEvent

class Watchdog:
    def __init__(self):
        self.anomaly_count = 0
        event_bus.subscribe("*", self.monitor_all)
        event_bus.subscribe("ANOMALY_DETECTED", self.handle_anomaly)
        print("[Watchdog] Initialized and monitoring for system anomalies.")

    async def monitor_all(self, event: AgentEvent):
        # Basic monitoring logic
        if event.type == "GATE_REJECTED":
            # print(f"[Watchdog] ALERT: Action rejected by gate! Reason: {event.payload.get('reason')}")
            self.anomaly_count += 1
        
        if self.anomaly_count > 5:
            print("[Watchdog] CRITICAL: Rejection threshold exceeded. Potential infinite loop or safety risk.")
            # In a real system, we might pause the agent here.

    async def handle_anomaly(self, event: AgentEvent):
        divergence = event.payload.get("divergence", 0)
        step_id = event.payload.get("step_id", "unknown")
        print(f"\n[Watchdog] ALERT: High divergence ({divergence:.2f}) detected for step {step_id}!")
        print(f"   └─ Predicted: {event.payload.get('expected', {}).get('predicted_outcome')}")
        print(f"   └─ Actual: {event.payload.get('actual')}")

# Global instance
watchdog = Watchdog()
