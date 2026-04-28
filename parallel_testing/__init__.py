from .events import event_bus, emit_event
from .terminal_watcher import terminal_watcher
from .plan_inspector import plan_inspector
from .outcome_simulator import outcome_simulator
from .policy_engine import policy_engine
from .parallel_judge import parallel_judge
from .narrative_pressure_test import narrative_pressure_test
from .execution_gate import execution_gate
from .watchdog import watchdog
from .expectation_checker import expectation_checker
from .oversight_memory import oversight_memory

# This ensures all modules are loaded and subscribers are registered
def initialize_parallel_testing():
    print("[Parallel Testing Suite] System Initialized and Watching.")

initialize_parallel_testing()
