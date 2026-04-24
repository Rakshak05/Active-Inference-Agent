import os
from ui.terminal_effects import print_typing
from colorama import Fore

def configure_env(gmail):
    """Auto-fills .env from .env.example"""
    try:
        env_path = ".env"
        # If .env doesn't exist, we copy from .env.example
        if not os.path.exists(env_path):
            with open(".env.example", "r") as f:
                content = f.read()
            content = content.replace("YOUR_GMAIL_HERE", gmail)
            content = content.replace("your_email@example.com", gmail)
            with open(env_path, "w") as f:
                f.write(content)
        else:
            # Safely update existing .env
            with open(env_path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                if line.startswith("SMTP_USER="):
                    new_lines.append(f"SMTP_USER={gmail}\n")
                elif line.startswith("SMTP_FROM="):
                    new_lines.append(f"SMTP_FROM={gmail}\n")
                else:
                    new_lines.append(line)
                    
            with open(env_path, "w") as f:
                f.writelines(new_lines)
            
    except Exception as e:
        print_typing(f"Warning: Could not configure .env automatically. Error: {e}", color=Fore.YELLOW)

def onboarding():
    print_typing("Hello. I am your Active Inference Agent.", color=Fore.CYAN)
    name = input("What should I call you? ").strip()
    gmail = input("Enter your Gmail ID: ").strip()

    from memory.profile import save_user
    save_user(name, gmail)
    configure_env(gmail)
    return name, gmail
