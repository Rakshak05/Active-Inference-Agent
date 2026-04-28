import asyncio
from .events import event_bus, AgentEvent

class TerminalWatcher:
    def __init__(self):
        self.logs = []
        # Subscribe to terminal output events
        event_bus.subscribe("TERMINAL_OUTPUT", self.handle_terminal_output)
        print("[TerminalWatcher] Watching agent terminal...")

    async def handle_terminal_output(self, event: AgentEvent):
        line = event.payload.get("line", "")
        source = event.payload.get("source", "unknown")
        # print(f"[TerminalWatcher] [{source}] {line}", end="")
        self.logs.append({"timestamp": event.timestamp, "line": line, "source": source})

    def get_logs(self):
        return self.logs

# Global instance
terminal_watcher = TerminalWatcher()
