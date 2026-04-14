import json
import os
import sys
from agent_manager import AgentManager

def test_real_active_inference():
    print("==================================================")
    print("LAUNCHING REAL ACTIVE INFERENCE SANDBOX TEST")
    print("==================================================")
    
    # Initialize the real continuous agent
    agent = AgentManager(efe_threshold=0.6, max_replans=3)

    # Allow user to assign tasks via CLI args or interactive input
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = input("\nEnter the task for the Active Inference Agent:\\n> ")
        
    if not task.strip():
        print("No task provided. Exiting.")
        return

    print(f"\nTask: {task}\n")
    
    # Run continuous Active Inference loop
    result = agent.process_task(user_instruction=task, max_steps=10)

    print("\n\n=== FINAL RESULT ===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_real_active_inference()
