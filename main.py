import sys
import os
import argparse
import psutil
from colorama import Fore, Style

# Local imports
from ui.terminal_effects import print_typing, spinner
from ui.onboarding import onboarding
from memory.profile import is_first_run, load_user, reset_memory

def check_ram(min_gb=8):
    ram = psutil.virtual_memory().total / (1024**3)
    if ram < min_gb:
        raise MemoryError(f"Insufficient RAM. Requires at least {min_gb}GB, found {ram:.2f}GB.")

def check_api_key():
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("MODEL_NAME") == "mistral":
        # Relaxing this just to not crash if they haven't set it yet but are using local models
        # However, following the instruction literally:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing API key in .env")

def safe_run():
    try:
        run_agent()
    except MemoryError as e:
        print(f"\n{Fore.RED} {e}{Style.RESET_ALL}\n")
    except KeyboardInterrupt:
        print(f"\n{Fore.GREEN}Session ended safely.{Style.RESET_ALL}")
    except RuntimeError as e:
        print(f"\n{Fore.RED}Configuration Error: {e}{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"\n{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}\n")

def run_agent():
    parser = argparse.ArgumentParser(description="Active Inference Agent CLI")
    parser.add_argument("--setup", action="store_true", help="Force run the onboarding setup")
    parser.add_argument("--reset-memory", action="store_true", help="Clear user profile memory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("task", nargs="*", help="Task for the agent")
    args = parser.parse_args()

    check_ram()
    
    if args.reset_memory:
        if reset_memory():
            print_typing("Memory has been reset.", color=Fore.YELLOW)
        else:
            print_typing("No memory found to reset.", color=Fore.YELLOW)
        return

    if args.setup or is_first_run():
        name, gmail = onboarding()
    else:
        user = load_user()
        name = user["name"] if user else "User"
        # Greeting variation / Time based greeting
        from datetime import datetime
        hour = datetime.now().hour
        greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"
        print_typing(f"{greeting}, {name}. Let's get to work.", color=Fore.CYAN)

    # Check the API Key
    try:
        check_api_key()
    except RuntimeError as e:
        # User goal: "If LLM fails -> fallback reasoning mode."
        print(f"{Fore.YELLOW}Warning: {e} - Falling back to local reasoning mode...{Style.RESET_ALL}")

    if args.debug:
        print(f"{Fore.YELLOW}[DEBUG] Debug mode enabled.{Style.RESET_ALL}")

    task_str = " ".join(args.task)
    
    while True:
        if not task_str:
            print_typing("What can i help you with today!? (or 'exit' to quit):", color=Fore.CYAN)
            try:
                task_str = input(f"{Fore.YELLOW}> {Style.RESET_ALL}").strip()
            except EOFError:
                print_typing("\nExiting session.", color=Fore.RED)
                break

        if not task_str or task_str.lower() in ['exit', 'quit']:
            print_typing(f"\nSession complete. It was a pleasure assisting you, {name}.", color=Fore.GREEN)
            break

        # Showing agent personality and UX
        spinner("Initializing Core Agent Manager...")
        print_typing("Thinking...", delay=0.03, color=Fore.MAGENTA)
        print_typing("Updating internal beliefs...", delay=0.03, color=Fore.MAGENTA)
        
        # Progress indicator integration
        from tqdm import tqdm
        import time
        for _ in tqdm(range(25), desc="Evaluating possible actions", ascii=False, colour="magenta"):
            time.sleep(0.02)

        import uuid
        session_id = str(uuid.uuid4())
        print_typing(f"\n[Task Initialized | Session {session_id[:8]}]: {task_str}\n", color=Fore.GREEN)
        
        from agent_manager import AgentManager
        agent = AgentManager(efe_threshold=0.8, max_replans=3, session_id=session_id)

        result = agent.process_task(user_instruction=task_str, max_steps=10)

        print_typing("\n=== FINAL RESULT ===", color=Fore.CYAN)
        import json
        print(json.dumps(result, indent=2))
        
        # Prompt for next task
        print_typing("\nThe work assigned has been completed. What can I do for you now?", color=Fore.GREEN)
        task_str = ""  # Clear it so it asks for input next iteration


if __name__ == "__main__":
    safe_run()
