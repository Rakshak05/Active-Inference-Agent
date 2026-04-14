# test_real_scenario.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agent_manager import AgentManager

# Initialize agent
agent = AgentManager(efe_threshold=0.5, max_replans=3)

# Test with your actual use case
result = agent.process_task(
    "Search for the latest research papers on active inference "
    "published in 2024 and create a summary of key findings"
)

print(f"\nStatus: {result['status']}")
print(f"Planning Attempts: {result['planning_attempts']}")

if result['status'] == 'success':
    print(f"Final EFE: {result['efe_analysis']['total_efe']:.4f}")
    print("\nReal scenario test PASSED")
else:
    print(f"\nTask failed: {result['reason']}")

# Export session log for analysis
agent.export_session_log("real_scenario_log.json")
print("\nSession log saved to: real_scenario_log.json")
