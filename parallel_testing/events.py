import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Callable, Awaitable
import json

@dataclass
class AgentEvent:
    type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_json(self):
        return json.dumps(self.__dict__)

class EventBus:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance.subscribers: Dict[str, List[Callable[[AgentEvent], Awaitable[None]]]] = {}
        return cls._instance

    def subscribe(self, event_type: str, callback: Callable[[AgentEvent], Awaitable[None]]):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        print(f"[EventBus] Subscribed to {event_type}")

    async def emit(self, event_type: str, payload: Dict[str, Any]):
        event = AgentEvent(type=event_type, payload=payload)
        # print(f"[EventBus] Emitting {event_type}")
        
        tasks = []
        # Notify specific subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                tasks.append(callback(event))
        
        # Notify 'ALL' subscribers
        if "*" in self.subscribers:
            for callback in self.subscribers["*"]:
                tasks.append(callback(event))
                
        if tasks:
            await asyncio.gather(*tasks)

# Global dispatcher instance
event_bus = EventBus()

async def emit_event(event_type: str, **kwargs):
    await event_bus.emit(event_type, kwargs)
